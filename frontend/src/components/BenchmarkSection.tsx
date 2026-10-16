import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { FlaskConical, Play, Loader2 } from 'lucide-react';
import { ExperimentResponse } from '../types/mission';
import { missionApi } from '../api/client';

export const BenchmarkSection: React.FC = () => {
  const [results, setResults] = useState<ExperimentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seedCount, setSeedCount] = useState(25);

  const handleRunBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await missionApi.runExperiments(seedCount, true);
      setResults(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark request failed.');
    } finally {
      setLoading(false);
    }
  };

  // Transform summary_table into grouped chart data per noise level
  const chartData = results
    ? results.noise_levels.map((noise) => {
        const rows = results.summary_table.filter((r) => r.noise_std === noise);
        const bayes = rows.find((r) => r.policy_name.includes('Bayesian'));
        const base = rows.find((r) => r.policy_name.includes('Baseline'));
        const rl = rows.find((r) => r.policy_name.includes('Q-Learning'));
        return {
          noise: `σ=${noise.toFixed(2)}`,
          'Bayesian + EIG (%)': bayes?.success_rate ?? 0,
          'Uninformed Baseline (%)': base?.success_rate ?? 0,
          'Tabular Q-Learning (%)': rl?.success_rate ?? 0,
          'Bayesian Avg Actions': bayes?.avg_actions_overall ?? 0,
          'Baseline Avg Actions': base?.avg_actions_overall ?? 0,
        };
      })
    : [];

  return (
    <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-5 shadow-xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2.5">
          <FlaskConical className="w-5 h-5 text-amber-400" />
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-100">
              Reproducible Multi-Seed Benchmark & Q-Learning Experiment
            </h2>
            <p className="text-xs text-slate-400">
              Evaluates Bayesian + EIG Autonomous Agent vs. Uninformed Exploration Baseline & Tabular Q-Learning across identical seeded worlds.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 font-mono">
            Seeds/Noise:
            <select
              value={seedCount}
              onChange={(e) => setSeedCount(Number(e.target.value))}
              className="ml-1.5 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
            >
              <option value={10}>10 seeds (Fast)</option>
              <option value={25}>25 seeds (Standard)</option>
              <option value={30}>30 seeds (Full)</option>
            </select>
          </label>

          <button
            onClick={handleRunBenchmark}
            disabled={loading}
            className="flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-semibold text-xs px-3.5 py-2 rounded-lg transition shadow"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Running {seedCount * 3} Trials...
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-slate-950" /> Run Live Benchmark
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-950/50 border border-rose-500/40 text-rose-200 text-xs p-3 rounded-lg">
          Benchmark Error: {error}
        </div>
      )}

      {!results && !loading && (
        <div className="text-xs text-slate-400 font-mono py-6 text-center bg-slate-950/50 rounded-lg border border-slate-800/60">
          Click "Run Live Benchmark" to execute real multi-seed simulations across 3 sensor noise levels (σ = 0.06, 0.10, 0.16) and plot comparative metrics.
        </div>
      )}

      {results && (
        <div className="space-y-4">
          {/* Interpretation Banner */}
          <div className="bg-cyan-950/30 border border-cyan-500/30 rounded-lg p-3 text-xs text-cyan-100 leading-relaxed">
            <strong className="text-cyan-300 uppercase font-mono mr-1.5">
              Measured Findings:
            </strong>
            {results.interpretation}
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 h-64">
              <div className="text-xs font-semibold text-slate-300 mb-2 font-mono">
                Mission Success Rate (%) by Sensor Noise Level
              </div>
              <ResponsiveContainer width="100%" height="88%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="noise" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="Bayesian + EIG (%)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Tabular Q-Learning (%)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Uninformed Baseline (%)" fill="#64748b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 h-64">
              <div className="text-xs font-semibold text-slate-300 mb-2 font-mono">
                Average Actions Taken per Mission (Lower is Better)
              </div>
              <ResponsiveContainer width="100%" height="88%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="noise" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      fontSize: '12px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="Bayesian Avg Actions" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Baseline Avg Actions" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                  <th className="py-2 px-3">Policy</th>
                  <th className="py-2 px-3">Noise (σ)</th>
                  <th className="py-2 px-3">Trials</th>
                  <th className="py-2 px-3">Success Rate</th>
                  <th className="py-2 px-3">Avg Actions (Win)</th>
                  <th className="py-2 px-3">Avg Energy</th>
                  <th className="py-2 px-3">Avg Scans</th>
                  <th className="py-2 px-3">Failed/Stranded</th>
                </tr>
              </thead>
              <tbody>
                {results.summary_table.map((row, idx) => (
                  <tr
                    key={`${row.policy_name}-${row.noise_std}-${idx}`}
                    className="border-b border-slate-800/50 hover:bg-slate-900/40"
                  >
                    <td className="py-2 px-3 font-semibold text-slate-200">
                      {row.policy_name}
                    </td>
                    <td className="py-2 px-3 text-teal-300">{row.noise_std.toFixed(2)}</td>
                    <td className="py-2 px-3">{row.trials}</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">
                      {row.success_rate.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3">{row.avg_actions_on_success.toFixed(1)}</td>
                    <td className="py-2 px-3">{row.avg_energy_consumed.toFixed(1)}</td>
                    <td className="py-2 px-3">{row.avg_scans.toFixed(1)}</td>
                    <td className="py-2 px-3 text-rose-400">
                      {row.stranded_or_failed_rate.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
