import React from 'react';
import { Star, Radio, Volume2, VolumeX, AlertTriangle } from 'lucide-react';
import { ChaosState, RadioMessage } from '../types/mission';

interface ChaosAndRadioPanelProps {
  chaos: ChaosState;
  latestCommentary: RadioMessage | null;
  commentaryLog: RadioMessage[];
  humorEnabled: boolean;
  onToggleHumor: () => void;
}

export const ChaosAndRadioPanel: React.FC<ChaosAndRadioPanelProps> = ({
  chaos,
  latestCommentary,
  commentaryLog,
  humorEnabled,
  onToggleHumor,
}) => {
  return (
    <div className="flex flex-col gap-3">
      {/* 5-Star Chaos Meter Card */}
      <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-4 shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Mission Stress / Chaos Level
            </span>
          </div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
            {chaos.status_label} ({chaos.raw_score} pts)
          </span>
        </div>

        {/* GTA-Style 5 Star Row */}
        <div className="flex items-center justify-between bg-slate-950/80 border border-slate-800/80 rounded-lg px-3 py-2.5 mb-2">
          <div className="flex items-center gap-1.5">
            {[1, 2, 3, 4, 5].map((starIdx) => {
              const active = starIdx <= chaos.stars;
              return (
                <Star
                  key={starIdx}
                  className={`w-5 h-5 transition-all duration-300 ${
                    active
                      ? 'text-amber-400 fill-amber-400 drop-shadow-[0_0_8px_rgba(245,158,11,0.65)] scale-105'
                      : 'text-slate-700 fill-slate-900'
                  }`}
                />
              );
            })}
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Rule-Driven Stress Index
          </span>
        </div>

        <p className="text-xs italic text-amber-200/90 mb-2">
          "{chaos.headline_quote}"
        </p>

        {chaos.recent_reasons.length > 0 && (
          <div className="border-t border-slate-800/80 pt-2 mt-1 space-y-1 max-h-24 overflow-y-auto">
            {chaos.recent_reasons.slice(0, 3).map((ev, i) => (
              <div
                key={`${ev.step}-${ev.event_code}-${i}`}
                className="flex items-center justify-between text-[11px] text-slate-400"
              >
                <span className="truncate pr-2">
                  <strong className="font-mono text-slate-300">T+{ev.step}:</strong> {ev.reason}
                </span>
                <span
                  className={`font-mono font-semibold shrink-0 ${
                    ev.delta_points >= 0 ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {ev.delta_points >= 0 ? `+${ev.delta_points}` : ev.delta_points}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Satirical Open-World Radio Transceiver Panel */}
      <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-4 shadow-lg flex flex-col gap-2.5">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-amber-400 animate-pulse" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Satirical Radio Comms (Ch. 94.2 FM)
            </h3>
          </div>
          <button
            onClick={onToggleHumor}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md border transition ${
              humorEnabled
                ? 'bg-amber-500/15 border-amber-500/40 text-amber-300 hover:bg-amber-500/25'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
            }`}
          >
            {humorEnabled ? (
              <>
                <Volume2 className="w-3.5 h-3.5" /> Humor ON
              </>
            ) : (
              <>
                <VolumeX className="w-3.5 h-3.5" /> Humor Muted
              </>
            )}
          </button>
        </div>

        {!humorEnabled ? (
          <div className="py-6 text-center text-xs text-slate-500 font-mono">
            [Satirical commentary disabled. Pure technical telemetry mode active.]
          </div>
        ) : (
          <>
            {latestCommentary && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <div className="flex items-center justify-between text-[11px] font-mono text-amber-300 mb-1">
                  <span>
                    {latestCommentary.speaker} • [{latestCommentary.event_type}]
                  </span>
                  <span>STEP {latestCommentary.step}</span>
                </div>
                <p className="text-sm font-medium text-slate-100 leading-snug">
                  "{latestCommentary.message}"
                </p>
                {latestCommentary.dispatcher_reply && (
                  <p className="text-xs text-teal-300/90 mt-1.5 font-mono">
                    ↳ DISPATCH: "{latestCommentary.dispatcher_reply}"
                  </p>
                )}
              </div>
            )}

            <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
              {commentaryLog.slice(1, 6).map((msg, idx) => (
                <div
                  key={`${msg.step}-${idx}`}
                  className="text-xs bg-slate-900/70 border border-slate-800/70 rounded px-2.5 py-1.5 text-slate-300"
                >
                  <span className="font-mono text-[10px] text-amber-400 mr-1.5">
                    [T+{msg.step} {msg.event_type}]
                  </span>
                  "{msg.message}"
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
