"""
Lightweight LeWAM inference server.

Receives context frames + state + task over TCP, runs inference, returns actions.
No lerobot async infrastructure needed.

Usage:
    python -m lerobot_policy_lewam.serve_lewam
    python -m lerobot_policy_lewam.serve_lewam --model ehalicki/lewam-so101-multitask-finetuned --device cuda --port 8080
"""

import argparse
import pickle
import socket
import struct
import time

import cv2
import numpy as np
import torch


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    raw_len = _recvall(sock, 4)
    length = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recvall(sock, length))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def load_model(repo_id, device):
    from lerobot_policy_lewam.modeling_lewam import LeWAMPolicy

    policy = LeWAMPolicy.from_pretrained(repo_id)
    policy.to(device)
    policy.eval()
    return policy


def decode_frames(frames_dict, device):
    """Dict of {cam: [jpeg_bytes, ...]} -> (1, N_cams, T, C, H, W) float tensor."""
    cams = []
    for cam in sorted(frames_dict):
        decoded = []
        for jpg in frames_dict[cam]:
            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        cams.append(torch.stack(decoded))
    return torch.stack(cams).unsqueeze(0).to(device)


def pca_rgb(tokens, T, patch_h, patch_w):
    """Convert (N, D) token sequence to (T, patch_h, patch_w, 3) PCA-RGB grid."""
    from sklearn.decomposition import PCA

    arr = tokens.float().cpu().numpy()
    if not np.isfinite(arr).all():
        return np.zeros((T, patch_h, patch_w, 3), dtype=np.float32)
    rgb = PCA(n_components=3).fit_transform(arr)
    rgb -= rgb.min(axis=0)
    rgb /= rgb.max(axis=0) + 1e-8
    return rgb.reshape(T, patch_h, patch_w, 3).astype(np.float32)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def infer(policy, frames, state_np, task, ode_steps=None, cfg_scale=1.0):
    device = next(policy.parameters()).device
    model = policy.lewam
    cfg = policy.config

    n_cams = frames.shape[1]
    crop_size = model.video_encoder.preprocessor.crop_size
    patch_size = model.VJEPA_PATCH_SIZE
    frame_latent_h = crop_size // patch_size
    frame_latent_w = (crop_size // patch_size) * n_cams
    if frame_latent_h != model.frame_latent_h or frame_latent_w != model.frame_latent_w:
        model.set_patch_grid(frame_latent_h, frame_latent_w, n_cams)

    context_tokens = model.encode_video(frames)

    state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
    norm_state = model.normalize_state(state)

    lang_tokens, lang_mask = None, None
    if model.vlm_encoder is not None and task:
        n_cams = frames.shape[1]
        last_frame = torch.cat([frames[:, i, -1] for i in range(n_cams)], dim=-1)
        lang_tokens, lang_mask = model.encode_language([task], images=last_frame)

    pred_vid, pred_actions = model.ode_solve(
        context_tokens,
        norm_state,
        lang_tokens,
        lang_mask,
        num_steps=ode_steps or cfg.num_ode_steps,
        smooth=cfg.smooth_actions,
        cfg_scale=cfg_scale,
    )

    rel_actions = model.unnormalize_actions(pred_actions)
    dt = 1.0 / cfg.action_fps
    abs_actions = state.unsqueeze(1) + torch.cumsum(rel_actions * dt, dim=1)

    future_viz = pca_rgb(
        pred_vid[0],
        model.num_future_tubelets,
        model.frame_latent_h,
        model.frame_latent_w,
    )

    return abs_actions[0, : cfg.n_action_steps].cpu().numpy(), future_viz


def main():
    p = argparse.ArgumentParser(description="LeWAM inference server")
    p.add_argument("--model", default="ehalicki/lewam-so101-multitask-finetuned")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    policy = load_model(args.model, args.device)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        policy.lewam = torch.compile(policy.lewam, dynamic=True)
        print("Model loaded and compiled.")
    else:
        print("Model loaded (no compile, GPU SM < 8.0).")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f"Listening on 0.0.0.0:{args.port}")

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Client connected: {addr}")
            try:
                while True:
                    msg = recv_msg(conn)
                    t0 = time.perf_counter()

                    frames = decode_frames(msg["frames"], args.device)
                    ode_steps = msg.get("ode_steps", policy.config.num_ode_steps)
                    cfg_scale = msg.get("cfg_scale", 1.0)
                    actions, future_viz = infer(policy, frames, msg["state"], msg["task"], ode_steps=ode_steps, cfg_scale=cfg_scale)

                    elapsed = time.perf_counter() - t0
                    print(f"Inference {elapsed:.2f}s  actions {actions.shape}")

                    send_msg(conn, {"actions": actions, "future_viz": future_viz})
            except ConnectionError:
                print(f"Client {addr} disconnected")
            finally:
                conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
