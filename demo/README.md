# Demos: studio and arcade

Live UIs for `/v1/systemone`. `systemone serve --demo` adds `/studio` (camera, motion,
audio) and `/arcade` (maze, snake) on the same host and port as the API.

## Setup

Install systemone from this package (the server itself is the standard library; the
engine is separate):

```bash
pip install -e ./systemone
```

Start vLLM or SGLang, then put systemone in front with `--demo`:

```bash
# vLLM already running, e.g. Gemma 4 31B:
systemone serve --upstream http://localhost:8000 --demo

# two servers, so the studio can pick a vision model and an audio model:
systemone serve --upstream http://localhost:8000 --upstream http://localhost:8001 --demo
```

Open `http://<host>:<port>/studio` and `/arcade` (default port 8011). Behind a token
proxy, add `?token=…`. Camera and microphone need an **HTTPS** page.

`--upstream` can be repeated: one endpoint can serve, say, Gemma 4 31B (vision, no
audio encoder) and Gemma 4 E4B (audio) from two vLLM servers. At startup the probe
checks whether each model takes images, video and audio; a model that can't returns
422 instead of guessing.

The request fields the pages use (also valid on `POST /v1/systemone` without the UI):

| field | effect |
|---|---|
| `images` | image data URLs |
| `video` | video data URLs |
| `video_frames` | JPEG frames; systemone packs them into an MJPEG clip in pure Python, so browsers need not encode video |
| `fps` | frame rate for `video_frames` |
| `audio` | WAV or other audio data URLs |

## Studio: camera, motion, audio

`/studio` classifies a **webcam or video file** (Camera), **short motion clips**
(Motion) and a **microphone or audio file** (Audio). Each tab has editable preset
questions and shows live probability bars, a history sparkline per question,
decisions/s and latency.

**Picking a model per tab** (Gemma 4, measured on one GPU; add your network RTT):

| task | Gemma 4 E4B | Gemma 4 31B |
|---|---|---|
| image: CIFAR-10 (10 classes) | 0.795 · **73 ms** · 32/s | **0.955** · 147 ms · 12.5/s |
| video: motion direction, 8-frame clips (4 classes) | 0.22 ⚠️ · 120 ms | **1.00** · 199 ms · 7/s |
| audio: spoken keywords, Speech Commands (10 words) | **0.84 · 72 ms · 78/s** | no audio encoder |
| audio: environmental sounds, ESC-10 (10 classes) | 0.36 · 197 ms · 26/s | no audio encoder |

(`/s` = items per second with 16 in flight.)

- **Camera and speech:** E4B, fast enough for several decisions a second.
- **Motion:** the 31B. E4B only ever answers left or right, never up or down, so it
  doesn't read motion.
- **Environmental sounds** are E4B's weak spot. It is strong on dogs, fire and crying
  babies (≥83%), but files most other noises under one label. Describing the classes
  made that worse.

The studio picks these defaults. Reproduce the numbers with:

```bash
pip install datasets numpy pillow soundfile librosa
python -m systemone.demo.media_eval --url http://127.0.0.1:8011 --models gemma4,gemma4-e4b --n 200
```

Raw run: [`../results/media-gemma4-2026-09-21.json`](../results/media-gemma4-2026-09-21.json).

## Arcade: maze and snake

`/arcade` plays **Maze** and **Snake** live, one `choice` question per move. You can
pick the model, the input (text board, image, or both), `think`, and hints, and see
each move's probabilities.

**Where the game runs matters more than the model.** In the default "server" mode, the
game loop runs inside systemone, next to the model (`POST /arcade/play`), and streams
one server-sent event per move. The viewer's network distance only delays the display.
In "browser" mode every move is a round trip to `/v1/systemone`, and the distance sets
the pace.

| from a laptop ~136 ms ping away | browser mode (round trip per move) | server mode (streamed) |
|---|---|---|
| text board | ~200 ms per move (~600 ms through an SSH tunnel) | **~82 ms (13–15 moves/s)** |
| image + text | ~400 ms | **~170 ms (6 moves/s)** |

Headless eval (same boards and prompts as the page):

```bash
python -m systemone.demo.arcade_eval --url http://127.0.0.1:8011 --model gemma4 --games 10
```

Gemma 4 31B (NVFP4), 10 games each:

| | Maze 7×7 solved (moves) | Snake 10×10 mean score (max) |
|---|---|---|
| random over legal moves | 0/10 | 3 (5) |
| scaffold heuristic (hints only) | 10/10 (55) | 34.8 (43) |
| Gemma 4, text + hints | 10/10 (**48**) | 30.8 (40) |
| Gemma 4, text, no hints | 0/10 | 17.4 (27) |
| Gemma 4, image + text + hints | 10/10 (48) | 26.5 (35) |
| Gemma 4, image only + hints | 10/10 (48) | 31.8 (43) |

One decision takes **~82 ms with a text board and ~150 ms with an image** on an idle
model. What these numbers show:

- **Maze:** the model solves every maze in fewer moves than the heuristic (48 vs 55),
  so it picks better turns.
- **Snake:** it plays reasonably from the board alone (17.4) but stays below the
  free-space heuristic (34.8). Snake mostly needs lookahead, which one read per move
  doesn't give.
- **Without hints** it loses track of where it has been and doesn't finish mazes.

Raw run: [`../results/arcade-gemma4-2026-09-21.json`](../results/arcade-gemma4-2026-09-21.json).

## Layout

| path | what |
|---|---|
| `web/studio.html` | camera / motion / audio UI |
| `web/arcade.html` | maze / snake UI (browser-driven mode has a JS copy of the board) |
| `arcade.py` | maze and snake: boards, prompts, PNG rendering, heuristic baselines |
| `play.py` | `POST /arcade/play` server-side game loop (SSE) |
| `media_eval.py` | image / video / audio accuracy and latency |
| `arcade_eval.py` | headless maze / snake vs random and heuristic |
