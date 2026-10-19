import React, { useEffect, useState, useRef } from 'react';
import {
  Play,
  Pause,
  StepForward,
  RotateCcw,
  FastForward,
  Download,
  BatteryCharging,
  Flame,
  Eye,
  EyeOff,
  Bot,
  AlertCircle,
  Clock,
  Compass,
} from 'lucide-react';
import { missionApi } from './api/client';
import {
  PublicMissionConfig,
  PublicMissionState,
  StepTraceEntry,
} from './types/mission';
import { GridMap } from './components/GridMap';
import { Scene3D } from './components/Scene3D';
import { ChaosAndRadioPanel } from './components/ChaosAndRadioPanel';
import { TelemetryAndTracePanel } from './components/TelemetryAndTracePanel';
import { BenchmarkSection } from './components/BenchmarkSection';

export const App: React.FC = () => {
  const [presets, setPresets] = useState<PublicMissionConfig[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('normal');
  const [mission, setMission] = useState<PublicMissionState | null>(null);
  const [trace, setTrace] = useState<StepTraceEntry[]>([]);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speedMs, setSpeedMs] = useState<number>(320);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [humorEnabled, setHumorEnabled] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('3d');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const timerRef = useRef<number | null>(null);

  // Initial load: fetch presets and create default mission
  useEffect(() => {
    const init = async () => {
      setLoading(true);
      try {
        const presetList = await missionApi.listPresets();
        setPresets(presetList);
        const initialMission = await missionApi.createMission('normal', true);
        setMission(initialMission);
        setTrace([]);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? `Backend Connection Error: ${err.message}. Ensure FastAPI server is running on port 8000.`
            : 'Unable to reach Python FastAPI backend.'
        );
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  // Auto-play loop
  useEffect(() => {
    if (!isPlaying || !mission) return;
    if (mission.status !== 'ACTIVE' && mission.status !== 'PAUSED') {
      setIsPlaying(false);
      return;
    }

    timerRef.current = window.setTimeout(async () => {
      try {
        const updated = await missionApi.stepMission(mission.mission_id, humorEnabled);
        setMission(updated);
        const updatedTrace = await missionApi.getTrace(mission.mission_id);
        setTrace(updatedTrace);
        if (updated.status !== 'ACTIVE') {
          setIsPlaying(false);
        }
      } catch (err) {
        setIsPlaying(false);
        setError(err instanceof Error ? err.message : 'Step execution failed.');
      }
    }, speedMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, mission, speedMs, humorEnabled]);

  const handleSelectPreset = async (presetId: string) => {
    setIsPlaying(false);
    setSelectedPreset(presetId);
    setLoading(true);
    try {
      const m = await missionApi.createMission(presetId, humorEnabled);
      setMission(m);
      setTrace([]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create mission.');
    } finally {
      setLoading(false);
    }
  };

  const handleStepOnce = async () => {
    if (!mission) return;
    setIsPlaying(false);
    try {
      const updated = await missionApi.stepMission(mission.mission_id, humorEnabled);
      setMission(updated);
      const updatedTrace = await missionApi.getTrace(mission.mission_id);
      setTrace(updatedTrace);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Step failed.');
    }
  };

  const handleFastComplete = async () => {
    if (!mission) return;
    setIsPlaying(false);
    setLoading(true);
    try {
      const updated = await missionApi.runMissionFast(mission.mission_id, humorEnabled, 120);
      setMission(updated);
      const updatedTrace = await missionApi.getTrace(mission.mission_id);
      setTrace(updatedTrace);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Fast simulation failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!mission) return;
    setIsPlaying(false);
    try {
      const updated = await missionApi.resetMission(mission.mission_id, humorEnabled);
      setMission(updated);
      setTrace([]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Reset failed.');
    }
  };

  const handleTogglePlayPause = async () => {
    if (!mission) return;
    if (isPlaying) {
      setIsPlaying(false);
      try {
        const paused = await missionApi.pauseMission(mission.mission_id);
        setMission(paused);
      } catch {
        // ignore pause error
      }
    } else {
      setIsPlaying(true);
    }
  };

  const handleDownloadTrace = () => {
    if (!mission) return;
    const payload = {
      mission_id: mission.mission_id,
      config: mission.config,
      final_status: mission.status,
      post_mission_evaluation: mission.post_mission_evaluation,
      trace,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `walle-trace-${mission.mission_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const batteryPct = mission
    ? Math.round((mission.battery / mission.battery_capacity) * 100)
    : 100;

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col">
      {/* Top Mission Control Header */}
      <header className="border-b border-slate-800/90 bg-[#0b1120]/95 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/15 border border-amber-400/40 flex items-center justify-center shadow-inner">
              <Bot className="w-6 h-6 text-amber-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold tracking-tight text-slate-100">
                  WALL-E: <span className="text-amber-400">NO SIGNAL, NO CHILL</span>
                </h1>
                {mission && (
                  <span
                    className={`text-[11px] font-mono font-bold px-2.5 py-0.5 rounded-full border ${
                      mission.status === 'SUCCESS'
                        ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300'
                        : mission.status === 'ACTIVE'
                        ? 'bg-amber-500/20 border-amber-400 text-amber-300'
                        : mission.status === 'PAUSED'
                        ? 'bg-slate-700 border-slate-500 text-slate-200'
                        : 'bg-rose-500/20 border-rose-400 text-rose-300'
                    }`}
                  >
                    {mission.status}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400">
                One missing robot. One unreliable sensor. One extremely underqualified hero.
              </p>
            </div>
          </div>

          {/* Mission Preset Selector & Primary Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedPreset}
              onChange={(e) => handleSelectPreset(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 font-medium focus:outline-none focus:border-amber-400"
            >
              {presets.map((p) => (
                <option key={p.preset_id} value={p.preset_id}>
                  {p.mission_name}
                </option>
              ))}
            </select>

            <button
              onClick={handleTogglePlayPause}
              disabled={!mission || (mission.status !== 'ACTIVE' && mission.status !== 'PAUSED')}
              className="flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-slate-950 font-bold text-xs px-3.5 py-2 rounded-lg transition shadow"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-3.5 h-3.5 fill-slate-950" /> Pause
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-slate-950" /> Auto-Run
                </>
              )}
            </button>

            <button
              onClick={handleStepOnce}
              disabled={
                !mission ||
                isPlaying ||
                (mission.status !== 'ACTIVE' && mission.status !== 'PAUSED')
              }
              className="flex items-center gap-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-100 font-semibold text-xs px-3 py-2 rounded-lg border border-slate-700 transition"
            >
              <StepForward className="w-3.5 h-3.5 text-cyan-400" /> Step
            </button>

            <button
              onClick={handleFastComplete}
              disabled={
                !mission ||
                isPlaying ||
                (mission.status !== 'ACTIVE' && mission.status !== 'PAUSED')
              }
              className="flex items-center gap-1 bg-teal-500/15 hover:bg-teal-500/25 disabled:opacity-40 text-teal-300 font-semibold text-xs px-3 py-2 rounded-lg border border-teal-500/30 transition"
              title="Complete mission immediately in backend"
            >
              <FastForward className="w-3.5 h-3.5" /> Fast Sim
            </button>

            <button
              onClick={handleReset}
              disabled={!mission}
              className="flex items-center gap-1 bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs px-2.5 py-2 rounded-lg border border-slate-800 transition"
              title="Reset Mission"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>

            <button
              onClick={handleDownloadTrace}
              disabled={!mission}
              className="flex items-center gap-1 bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs px-2.5 py-2 rounded-lg border border-slate-800 transition"
              title="Download JSON Action Trace"
            >
              <Download className="w-3.5 h-3.5 text-amber-400" /> Trace JSON
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl w-full mx-auto px-4 py-4 flex-1 space-y-4">
        {error && (
          <div className="bg-rose-950/70 border border-rose-500/50 rounded-xl p-3.5 flex items-center gap-3 text-xs text-rose-200">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {mission && (
          <>
            {/* Top Status & Controls Ribbon */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {/* Battery Gauge */}
              <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="flex items-center gap-1.5 font-semibold text-slate-300">
                    <BatteryCharging className="w-4 h-4 text-amber-400" /> Solar Battery
                  </span>
                  <span className="font-mono font-bold text-slate-200">
                    {mission.battery}/{mission.battery_capacity} ({batteryPct}%)
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden mt-2 border border-slate-800">
                  <div
                    className={`h-full transition-all duration-300 ${
                      batteryPct <= 22
                        ? 'bg-rose-500'
                        : batteryPct <= 50
                        ? 'bg-amber-400'
                        : 'bg-emerald-400'
                    }`}
                    style={{ width: `${Math.max(0, Math.min(100, batteryPct))}%` }}
                  />
                </div>
              </div>

              {/* Action Budget & Elapsed Simulation Steps */}
              <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="flex items-center gap-1.5 font-semibold text-slate-300">
                    <Clock className="w-4 h-4 text-cyan-400" /> Action Budget
                  </span>
                  <span className="font-mono font-bold text-slate-200">
                    {mission.step_count} / {mission.max_actions}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-400 mt-1">
                  Scans: <strong className="text-teal-300">{mission.metrics.total_scans}</strong> •
                  Explored: <strong className="text-slate-200">{mission.metrics.coverage_pct}%</strong>
                </div>
              </div>

              {/* Current Objective */}
              <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-3 flex flex-col justify-between">
                <div className="text-[10px] uppercase tracking-wider text-slate-400 flex items-center gap-1">
                  <Compass className="w-3.5 h-3.5 text-amber-400" /> Active Objective
                </div>
                <div className="text-xs font-mono font-bold text-amber-300 truncate mt-1">
                  {mission.current_objective}
                </div>
              </div>

              {/* Speed & Heatmap Controls */}
              <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-3 flex items-center justify-between col-span-2">
                <div className="flex items-center gap-2">
                  <Flame className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs text-slate-300 font-medium">Sim Speed:</span>
                  {[
                    { label: 'Slow', ms: 600 },
                    { label: 'Normal', ms: 300 },
                    { label: 'Turbo', ms: 110 },
                  ].map((opt) => (
                    <button
                      key={opt.label}
                      onClick={() => setSpeedMs(opt.ms)}
                      className={`text-xs font-mono px-2 py-1 rounded border transition ${
                        speedMs === opt.ms
                          ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => setShowHeatmap((prev) => !prev)}
                  className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition ${
                    showHeatmap
                      ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  {showHeatmap ? (
                    <>
                      <Eye className="w-3.5 h-3.5" /> Heatmap ON
                    </>
                  ) : (
                    <>
                      <EyeOff className="w-3.5 h-3.5" /> Heatmap OFF
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Primary 2-Column Mission Control Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
              {/* Left Column (5 cols): Interactive Grid Map + Chaos & Radio */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex items-center justify-end gap-1.5">
                  {(['3d', '2d'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setViewMode(m)}
                      className={`text-xs font-mono px-3 py-1 rounded border transition ${
                        viewMode === m
                          ? 'bg-amber-500/20 border-amber-400 text-amber-300'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {m === '3d' ? '3D View' : '2D Grid'}
                    </button>
                  ))}
                </div>
                {viewMode === '3d' ? (
                  <Scene3D mission={mission} showHeatmap={showHeatmap} />
                ) : (
                  <GridMap mission={mission} showHeatmap={showHeatmap} />
                )}
                <ChaosAndRadioPanel
                  chaos={mission.chaos}
                  latestCommentary={mission.latest_commentary}
                  commentaryLog={mission.commentary_log}
                  humorEnabled={humorEnabled}
                  onToggleHumor={() => setHumorEnabled((h) => !h)}
                />
              </div>

              {/* Right Column (7 cols): Bayesian Telemetry, Explainable Trace & Final Report */}
              <div className="lg:col-span-7 space-y-4">
                <TelemetryAndTracePanel mission={mission} trace={trace} />
              </div>
            </div>

            {/* Bottom Benchmark & Reinforcement Learning Section */}
            <BenchmarkSection />
          </>
        )}

        {loading && !mission && (
          <div className="py-20 text-center text-sm text-slate-400 font-mono">
            Initializing WALL-E Autonomous Search & Rescue Telemetry...
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
