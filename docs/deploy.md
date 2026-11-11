# Putting sky-clock on the web

The app runs as one Docker image (the repository's `Dockerfile`, with local plate solving) in public mode
(`SKYCLOCK_PUBLIC=1`):

* searches are limited to 6 h windows and at most 2 analyses run at once;
* the Space-Track login is hidden (nobody types passwords into a shared site);
* every mode asks for the responsible-use confirmation;
* photos stay in memory and are never stored (see the About page).

It needs about **1 GB of RAM** (measured: 540 MB peak for one analysis with a small catalogue, inside the
container; a full CelesTrak catalogue and two users at once need more). That rules out 512 MB free tiers.

| Host | Cost | Notes (checked 2026-11-11) |
|---|---|---|
| **Hugging Face Space** (Docker) | PRO account, $9/month | 2 vCPU, 16 GB RAM. Hugging Face's docs say Docker Spaces "require a paid plan to create". Auto-deploy from GitHub is set up below. Sleeps when unused. |
| Google Cloud Run | free tier: 240,000 vCPU-seconds and 450,000 GiB-seconds per month | Needs a Google Cloud account with billing. Scales to zero; a cold start takes a while (large image). |
| Render free web service | $0 | 512 MB RAM: too small. Spins down after 15 min idle. |
| Your own laptop + `docker compose up` | $0 | Only reachable while it runs; fine for yourself and friends via a tunnel. |

Sources: huggingface.co/docs/hub/spaces-overview, huggingface.co/pricing, render.com/docs/free,
render.com/pricing, cloud.google.com/run/pricing (read 2026-11-11; prices change).

## Hugging Face Space: one-time setup (done by the repository owner, about 10 minutes)

1. Create an account at <https://huggingface.co/join> (accept their terms yourself) and subscribe to PRO at
   <https://huggingface.co/subscribe/pro>.
2. Create a Space: <https://huggingface.co/new-space>
   * Space name: `sky-clock` (or any name);
   * licence: MIT;
   * **Space SDK: Docker**, template **Blank**;
   * hardware: **CPU basic (free)**;
   * visibility: Public (or Private while you try it).
3. Create a token: <https://huggingface.co/settings/tokens> → *Create new token* → type **Write**
   (or a fine-grained token with write access to that one Space only). Copy it.
4. In the GitHub repository: *Settings → Secrets and variables → Actions*:
   * *Secrets* tab → *New repository secret*: name `HF_TOKEN`, value: the token;
   * *Variables* tab → *New repository variable*: name `HF_SPACE`, value `your-hf-username/sky-clock`.
5. Start the first deployment: *Actions → deploy-space → Run workflow*. After that, every merge to `main` that
   passes the tests is deployed automatically.
6. Open `https://huggingface.co/spaces/your-hf-username/sky-clock`. The first build takes about 10 minutes
   (it installs astrometry.net and its star index).

Nothing secret is stored in the repository: the token lives only in GitHub's encrypted secrets.

## Limits you can change

Set these as *Variables* in the Space's settings (Settings → Variables and secrets):

| Variable | Default (public) | Meaning |
|---|---|---|
| `SKYCLOCK_MAX_WINDOW_H` | 6 | longest search window, hours |
| `SKYCLOCK_MAX_CONCURRENT` | 2 | analyses running at the same time |

## Google Cloud Run (free tier)

1. Create a Google Cloud account and a project at <https://console.cloud.google.com/> (accept their terms and
   add billing yourself; the free tier covers light use).
2. Install the gcloud CLI on your laptop and run `gcloud auth login`.
3. From the repository folder:

   ```bash
   gcloud run deploy sky-clock --source . --region europe-north1 --memory 2Gi --cpu 2 \
       --port 8501 --set-env-vars SKYCLOCK_PUBLIC=1 --allow-unauthenticated --max-instances 1
   ```

   `--max-instances 1` caps the bill: one container at most, whatever the traffic.

## Other hosts

Any host that runs a Docker image with about 1 to 2 GB of RAM works (a small VPS, Fly.io): build the
`Dockerfile`, set `SKYCLOCK_PUBLIC=1` and `PORT`. Streamlit Community Cloud is not suitable: it cannot install
astrometry.net, so plate solving would need nova.astrometry.net.
