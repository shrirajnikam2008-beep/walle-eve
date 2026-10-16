import React from 'react';
import {
  Bot,
  Zap,
  Radio,
  Crosshair,
  ShieldAlert,
  Sparkles,
  BrickWall,
} from 'lucide-react';
import { Coordinate, PublicMissionState } from '../types/mission';

interface GridMapProps {
  mission: PublicMissionState;
  showHeatmap: boolean;
}

function coordKey(c: Coordinate | number[]): string {
  return `${c[0]},${c[1]}`;
}

export const GridMap: React.FC<GridMapProps> = ({ mission, showHeatmap }) => {
  const {
    config,
    walle_position,
    explored_cells,
    known_obstacles,
    scan_history,
    belief_map,
    most_likely_cell,
    max_confidence,
    planned_route,
    post_mission_evaluation,
  } = mission;

  const width = config.grid_width;
  const height = config.grid_height;

  const exploredSet = new Set(explored_cells.map(coordKey));
  const obstacleSet = new Set(known_obstacles.map(coordKey));
  const chargerSet = new Set(config.charging_stations.map(coordKey));
  const routeSet = new Set(planned_route.slice(1).map(coordKey));

  const scanMap = new Map<string, number>();
  for (const s of scan_history) {
    scanMap.set(coordKey(s.position), s.signal);
  }

  const trueTargetKey =
    post_mission_evaluation?.true_target_position != null
      ? coordKey(post_mission_evaluation.true_target_position)
      : null;
  const trueDecoySet = new Set(
    (post_mission_evaluation?.true_decoy_positions || []).map(coordKey)
  );

  // Normalize belief color intensity relative to current max belief
  const peakProb = Math.max(max_confidence, 0.01);

  return (
    <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-4 shadow-xl flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-amber-400" />
          <h2 className="text-sm font-semibold tracking-wider uppercase text-slate-200">
            Sector Tactical Grid ({width} × {height})
          </h2>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-teal-300 border border-teal-500/20">
            Seed #{config.seed}
          </span>
        </div>

        {/* Map Legend */}
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-400 inline-block" /> WALL-E
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm border border-cyan-400 bg-cyan-500/20 inline-block" /> MAP Estimate
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/30 border border-emerald-400 inline-block" /> Solar Dock
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-slate-700 inline-block" /> Obstacle
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400/70 inline-block" /> A* Route
          </span>
        </div>
      </div>

      {/* 12x12 Grid Container */}
      <div
        className="grid gap-1 w-full max-w-[580px] mx-auto aspect-square select-none"
        style={{
          gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`,
        }}
      >
        {Array.from({ length: height }).map((_, y) =>
          Array.from({ length: width }).map((__, x) => {
            const key = `${x},${y}`;
            const isWalle = walle_position[0] === x && walle_position[1] === y;
            const isObstacle = obstacleSet.has(key);
            const isCharger = chargerSet.has(key);
            const isExplored = exploredSet.has(key);
            const isMostLikely =
              most_likely_cell[0] === x && most_likely_cell[1] === y;
            const isRoute = routeSet.has(key);
            const scanVal = scanMap.get(key);
            const prob = belief_map[x]?.[y] ?? 0;
            const isTrueTarget = trueTargetKey === key;
            const isDecoy = trueDecoySet.has(key);

            const relIntensity = showHeatmap && !isObstacle ? Math.min(1, prob / peakProb) : 0;
            const heatBg =
              showHeatmap && !isObstacle && prob > 0.001
                ? `rgba(6, 182, 212, ${(0.06 + relIntensity * 0.58).toFixed(2)})`
                : undefined;

            return (
              <div
                key={key}
                title={`Cell (${x}, ${y}) | Posterior P=${(prob * 100).toFixed(2)}%${
                  scanVal !== undefined ? ` | Last Scan=${scanVal.toFixed(3)}` : ''
                }`}
                style={{ backgroundColor: heatBg }}
                className={`relative rounded-md border flex flex-col items-center justify-center transition-all duration-200 overflow-hidden ${
                  isObstacle
                    ? 'bg-slate-900/95 border-slate-700/80 text-slate-600'
                    : isWalle
                    ? 'border-amber-400 ring-2 ring-amber-400/60 bg-amber-500/20 z-10'
                    : isTrueTarget
                    ? 'border-emerald-400 ring-2 ring-emerald-400/60 bg-emerald-500/25'
                    : isMostLikely
                    ? 'border-cyan-400 ring-1 ring-cyan-400/50 bg-cyan-950/40'
                    : isExplored
                    ? 'border-slate-700/70 bg-slate-900/40'
                    : 'border-slate-800/70 bg-[#080d19]'
                }`}
              >
                {/* Coordinate watermark on corners or hover */}
                <span className="absolute top-0.5 left-1 text-[8px] font-mono text-slate-500/70 leading-none">
                  {x},{y}
                </span>

                {/* Charger Badge */}
                {isCharger && !isWalle && (
                  <Zap className="w-3.5 h-3.5 text-emerald-400 opacity-85" />
                )}

                {/* Obstacle Icon */}
                {isObstacle && <BrickWall className="w-3.5 h-3.5 text-slate-600" />}

                {/* Planned A* route waypoint dot */}
                {isRoute && !isWalle && !isObstacle && (
                  <span className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_#f59e0b] animate-pulse" />
                )}

                {/* Most Likely Hypothesis Crosshair */}
                {isMostLikely && !isWalle && !isTrueTarget && (
                  <Crosshair className="w-3.5 h-3.5 text-cyan-300 animate-pulse" />
                )}

                {/* Post-mission True EVE Location */}
                {isTrueTarget && !isWalle && (
                  <div className="flex flex-col items-center">
                    <Sparkles className="w-4 h-4 text-emerald-300 animate-bounce" />
                    <span className="text-[8px] font-bold text-emerald-200 leading-none">
                      EVE
                    </span>
                  </div>
                )}

                {/* Post-mission Revealed Decoy Location */}
                {isDecoy && (
                  <div className="flex flex-col items-center" title="Revealed Decoy Transponder">
                    <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                    <span className="text-[7px] font-bold text-rose-300 leading-none">
                      DECOY
                    </span>
                  </div>
                )}

                {/* WALL-E Robot Avatar */}
                {isWalle && (
                  <div className="flex flex-col items-center justify-center">
                    <Bot className="w-5 h-5 text-amber-400 drop-shadow-[0_0_6px_rgba(245,158,11,0.8)]" />
                  </div>
                )}

                {/* Scan Marker Badge */}
                {scanVal !== undefined && !isWalle && (
                  <span className="absolute bottom-0.5 right-0.5 flex items-center text-[8px] font-mono px-1 rounded bg-slate-950/80 text-teal-300 border border-teal-500/30">
                    <Radio className="w-2 h-2 mr-0.5 text-teal-400" />
                    {(scanVal * 100).toFixed(0)}
                  </span>
                )}

                {/* Belief Percentage in heatmap mode for high-probability cells */}
                {showHeatmap && !isObstacle && !isWalle && prob >= 0.03 && scanVal === undefined && (
                  <span className="absolute bottom-0.5 text-[8px] font-mono text-cyan-200/90 leading-none">
                    {(prob * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
