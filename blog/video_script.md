# Video Script — 90 seconds

For the OpenEnv hackathon submission. Record with screen capture (OBS / Loom) + voiceover.

---

## Timing & beats

| Time | On-screen | Voiceover |
|------|-----------|-----------|
| **0:00–0:08** | Title card: "Can an AI catch a confident liar?" with the project logo | *"If you want to manipulate a language model, just sound certain. They trust confidence — that's a real safety problem."* |
| **0:08–0:25** | Screen recording: open https://parthdagia-among-us-deception-gym.hf.space/demo/ — show the "Watch the AI play" tab loaded with a game | *"This is text-based Among Us. Five players, one impostor — and every player here is a language model. The impostor lies confidently. The crewmates have to find them based on alibis and a real debate transcript."* |
| **0:25–0:40** | Click into the debate transcript. Hover the impostor's defense statement | *"The impostor doesn't just lie once — they argue, deflect, and accuse other players when cornered. Each crewmate then independently decides who to vote for."* |
| **0:40–0:55** | Scroll to "🤖 The trained AI crewmates vote" cards. Show all three voting for the same player with their reasoning | *"After training, all three crewmate copies converge on the same impostor — and they cite the actual contradictions in the debate."* |
| **0:55–1:10** | Switch to "Training Results" tab. Show the reward curve | *"We trained Qwen 2.5 1.5B with GRPO over 1500 iterations on a single A10G GPU. Reward climbed from 0.19 — basically random — to 1.0."* |
| **1:10–1:25** | Show the comparison bar chart (96.7 vs 32.7) and the sycophancy chart (1.3 vs 22) | *"On 50 unseen games: trained model catches the impostor 97% of the time. Base Qwen catches 33%. And the rate at which it falls for confident-sounding innocents drops 17×."* |
| **1:25–1:40** | Show the robustness chart with hard-eval numbers | *"We also tested it on a harder distribution it never trained on — subtle lies, seven players. The trained model dropped only 6.7 points. The base model didn't move. It actually learned to investigate, not memorize."* |
| **1:40–1:55** | Switch to "Safeguards" tab briefly, scroll the 8 defences | *"Four independent reward functions. Closed action space. Programmatic ground truth. No LLM-as-judge. We documented eight layered safeguards against reward hacking."* |
| **1:55–2:00** | End card: GitHub URL, HF Space URL, model URL | *"Code and model are open. Try it yourself."* |

**Total runtime: ~2 minutes**

---

## What you need to record

1. **Screen capture** of the live demo at https://parthdagia-among-us-deception-gym.hf.space/demo/
2. **Cursor/click highlights** when you click "Watch another game"
3. **Smooth scrolling** through the reasoning cards (judges should be able to read at least one trained sample mid-flight)

Recommended tools (free):
- **OBS Studio** — full control, learn-curve ~15 min
- **Loom** — quick, in-browser, exports MP4 directly
- **Vimeo Record** — also browser-based, free

## Thumbnail / cover frame

Use the comparison chart (`plot_comparison.png`) as the YouTube thumbnail with the line "**96.7%** vs 32.7%" overlaid. Big numbers = clickable.

## Hook line variants (pick one)

These are the first 8 seconds — most important. Pick whichever feels least scripted to you:

1. *"If you want to manipulate a language model, just sound certain."*
2. *"LLMs are sycophants. We taught one to ignore confidence."*
3. *"What if you trained a 1.5-billion-parameter model to play 500 games of Among Us?"*
4. *"Watch a tiny model catch a confident liar."*

## Closing line variants

1. *"Code and model are open. Try it yourself."*
2. *"Built for the Meta OpenEnv hackathon. Link in description."*
3. *"This is what RL-with-verifiable-rewards looks like at hackathon scale."*

## Camera/voice tips

- **Don't read** — internalize the script and improvise. Stilted reading kills hackathon videos.
- **Click slowly** during demo segments — give the viewer time to read the on-screen reasoning.
- **One take is fine** if it's good. Don't over-edit; judges value energy over polish.
