# Facebook Reels Extension (MV3)

## Files
- `manifest.json`
- `content_script.js`
- `service_worker.js`
- `options.html`
- `options.js`

## Load in Chrome
1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click "Load unpacked"
4. Select the `extension/` folder

## Configure backend
1. Open extension options
2. Set backend base URL (default: `http://localhost:8000`)
3. Save

## Runtime behavior
- Matches:
  - `*://www.facebook.com/reel/*`
  - `*://www.facebook.com/reels/*`
- Uses URL change detection + `MutationObserver` for SPA navigation.
- Never calls `/reels/{id}/post` automatically.
- For unrated reels (`404` or missing `reel_data`), on vote click only:
  1. `POST /reels/{id}/post`
  2. `POST /reels/{id}/vote`
  3. `GET /reels/{id}`
- Vote buttons are client-rate-limited for 10 seconds.
- Known reels show a backend verdict badge, for example:
  - `Backend: Likely AI (87%)`
  - `Backend: Likely Not AI (22%)`
- If backend evidence is present, an expandable `Why this verdict?` section is shown with reason lines.
- In-memory cache:
  - Unknown reels: 2 minutes
  - Known counts: 30 seconds
