"""Self-contained local report with interactive per-player occupancy maps."""

import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def write_report(folder: Path, video: Path, players: list[dict], teams: dict, benchmark: dict,
                 heatmaps: bool) -> None:
    data = json.dumps({"players": players, "teams": teams, "benchmark": benchmark,
                       "heatmaps": heatmaps}, allow_nan=False).replace("<", "\\u003c")
    video_url = quote(Path(os.path.relpath(video, folder)).as_posix(), safe="/")
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SoccerVision match report</title>
<style>
body{font:16px system-ui,sans-serif;background:#101820;color:#ecf2f8;max-width:1100px;margin:auto;padding:28px}
h1{font-size:2.3rem;margin-bottom:4px}p{line-height:1.6;color:#b9c8d8}a{color:#65c8ff}
.cards{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}.card{background:#1c2b39;padding:18px;border-radius:10px;flex:1;min-width:160px}
strong{font-size:1.5rem;display:block}label,select{font-size:1rem}select{padding:8px;margin:12px;background:#263d50;color:white}
img{max-width:100%;border-radius:8px}table{width:100%;border-collapse:collapse;margin-top:24px;font-size:.9rem}
th,td{text-align:left;padding:10px;border-bottom:1px solid #334655}th{color:#65c8ff}.table{overflow:auto}
</style><h1>SoccerVision</h1><p>Observed match trajectories and approximate image-space analytics.</p>
<p><a href="VIDEO">Open annotated MP4</a> · <a href="tracks.csv">Tracking CSV</a> · <a href="benchmark.json">Benchmark JSON</a></p>
<div class="cards" id="cards"></div>
<p id="possession"></p>
<label for="track">Player heatmap</label><select id="track"></select><p id="detail"></p><img id="map" alt="Selected player image-coordinate occupancy map">
<div class="table"><table><thead><tr><th>ID</th><th>Class</th><th>Team</th><th>Distance (px)</th><th>Mean speed (px/s)</th><th>Top speed (px/s)</th></tr></thead><tbody id="rows"></tbody></table></div>
<p>Pixel motion includes camera movement. Team labels are anonymous color clusters. Possession uses observed ball detections and a distance heuristic; unknown frames are excluded from team percentages. Tracks are not player identities across camera cuts.</p>
<script type="application/json" id="data">DATA</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent), b=d.benchmark, t=d.teams;
const number=v=>v==null?'Unknown':Number(v).toFixed(1);
for(const [label,value] of [['Frames',b.processed_frames],['Tracks',b.unique_tracks],['Pipeline FPS',number(b.pipeline_fps)],['Ball observed',`${t.ball_observed_frames}/${t.total_frames}`]]){
 const card=document.createElement('div');card.className='card'; const title=document.createElement('span');title.textContent=label;
 const val=document.createElement('strong');val.textContent=value;card.append(title,val);document.getElementById('cards').append(card);
}
document.getElementById('possession').textContent=`Controlled frames: ${t.controlled_frames}; unknown: ${t.unknown_frames}. Team 1: ${number(t.possession_percent['0'])}%; Team 2: ${number(t.possession_percent['1'])}%.`;
const select=document.getElementById('track');
for(const p of d.players){const option=document.createElement('option');option.value=p.track_id;option.textContent=`${p.class} #${p.track_id}`;select.append(option);
const row=document.createElement('tr');for(const v of [p.track_id,p.class,p.team==null?'Unknown':p.team+1,number(p.distance_px),number(p.average_speed_px_s),number(p.top_speed_px_s)]){const cell=document.createElement('td');cell.textContent=v;row.append(cell);}document.getElementById('rows').append(row);}
function show(){const p=d.players.find(p=>String(p.track_id)===select.value);if(!p)return;document.getElementById('detail').textContent=`Track ${p.track_id}: ${number(p.observed_seconds)} seconds of consecutive observed motion.`;const img=document.getElementById('map');img.hidden=!d.heatmaps;if(d.heatmaps)img.src=`plots/player_${p.track_id}_heatmap.png`;}
select.addEventListener('change',show);show();
</script></html>"""
    document = document.replace("VIDEO", html.escape(video_url, quote=True)).replace("DATA", data)
    (folder / "report.html").write_text(document, encoding="utf-8")
