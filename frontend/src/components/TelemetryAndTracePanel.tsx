import React from 'react';
import {
  BrainCircuit,
  Compass,
  ShieldCheck,
  Activity,
  CheckCircle2,
  AlertOctagon,
} from 'lucide-react';
import { PublicMissionState, StepTraceEntry } from '../types/mission';

interface TelemetryAndTracePanelProps {
  mission: PublicMissionState;
  trace: StepTraceEntry[];
}

export const TelemetryAndTracePanel: React.FC<TelemetryAndTracePanelProps> = ({
  mission,
  trace,
}) => {
  const {
    latest_explanation,
    most_likely_cell,
    max_confidence,
    entropy,
    inferred_eve_status,
    eve_status_belief,
    target_verified,
  } = mission;

  return (
    <div className="flex flex-col gap-3">
      {/* Bayesian Inference & Planner Telemetry Card */}
      <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-4 shadow-lg">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-3">
          <div className="flex items-center gap-2">
            <BrainCircuit className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Bayesian Inference & Autonomous Planner
            </h3>
          </div>
          <span
            className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
              target_verified
                ? 'bg-emerald-500/15 border-emerald-400 text-emerald-300'
                : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
            }`}
          >
            {target_verified ? 'VERIFIED TARGET CONTACT' : 'UNVERIFIED POSTERIOR ESTIMATE'}
          </span>
        </div>

        {/* Key Probabilistic Metrics */}
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-2.5">
            <div className="text-[10px] uppercase text-slate-400">MAP Hypothesis</div>
            <div className="text-base font-mono font-bold text-cyan-300">
              ({most_likely_cell[0]}, {most_likely_cell[1]})
            </div>
            <div className="text-[11px] text-slate-400">
              Confidence: <strong className="text-slate-200">{(max_confidence * 100).toFixed(1)}%</strong>
            </div>
          </div>

          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-2.5">
            <div className="text-[10px] uppercase text-slate-400">Shannon Entropy H(B)</div>
            <div className="text-base font-mono font-bold text-amber-300">
              {entropy.toFixed(3)} <span className="text-xs font-normal">nats</span>
            </div>
            <div className="text-[11px] text-slate-400">
              EIG: <strong className="text-slate-200">{(latest_explanation?.expected_entropy_reduction ?? 0).toFixed(3)}</strong>
            </div>
          </div>

          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-2.5">
            <div className="text-[10px] uppercase text-slate-400">Inferred EVE Status</div>
            <div className="text-xs font-mono font-bold text-teal-300 truncate mt-0.5">
              {inferred_eve_status}
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              P(Status):{' '}
              <strong className="text-slate-200">
                {((eve_status_belief[inferred_eve_status] ?? 0.33) * 100).toFixed(0)}%
              </strong>
            </div>
          </div>
        </div>

        {/* Technical Decision Rationale */}
        {latest_explanation ? (
          <div className="bg-slate-950/90 border border-cyan-500/25 rounded-lg p-3 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-1 text-xs">
              <span className="font-mono font-semibold text-cyan-300 flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5" />
                Action: {latest_explanation.chosen_action} → ({latest_explanation.target_cell[0]},{' '}
                {latest_explanation.target_cell[1]})
              </span>
              <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                Mode: {latest_explanation.objective_mode} | Risk: {latest_explanation.risk_assessment}
              </span>
            </div>
            <p className="text-xs text-slate-300 font-mono leading-relaxed">
              {latest_explanation.technical_rationale}
            </p>
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
              <span>
                Route Steps: <strong className="text-slate-200">{latest_explanation.travel_cost_steps}</strong>
              </span>
              <span>
                Est. Route Energy: <strong className="text-slate-200">{latest_explanation.estimated_energy_required}</strong>
              </span>
              <span>
                Charger Margin: <strong className="text-emerald-300">{latest_explanation.safety_margin_energy} units</strong>
              </span>
              {latest_explanation.replanned_due_to_evidence && (
                <span className="text-amber-300 font-semibold">⚡ Replanned on new evidence</span>
              )}
            </div>
          </div>
        ) : (
          <div className="text-xs text-slate-400 font-mono py-3 text-center">
            Ready at starting coordinates ({mission.walle_position[0]}, {mission.walle_position[1]}). Press Step or Auto-Run to begin autonomous loop.
          </div>
        )}
      </div>

      {/* Explainable Step Trace distinguishing Observed Facts, Inferred Conclusions & Uncertainty */}
      <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-4 shadow-lg flex flex-col gap-2">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-teal-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Explainable Decision & Observation Trace ({trace.length} steps)
            </h3>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Facts vs. Inferences vs. Uncertainty
          </span>
        </div>

        {trace.length === 0 ? (
          <div className="text-xs text-slate-500 font-mono py-6 text-center">
            No actions executed yet. Step through the simulation to inspect structured trace entries.
          </div>
        ) : (
          <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
            {trace
              .slice()
              .reverse()
              .slice(0, 12)
              .map((entry) => (
                <div
                  key={entry.step}
                  className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 text-xs space-y-1"
                >
                  <div className="flex items-center justify-between font-mono">
                    <span className="text-amber-300 font-semibold">
                      #{entry.step} {entry.action} ({entry.from_position[0]},{entry.from_position[1]}) → (
                      {entry.to_position[0]},{entry.to_position[1]})
                    </span>
                    <span className="text-slate-400">
                      Batt: {entry.battery_before}→{entry.battery_after} ({entry.energy_delta >= 0 ? `+${entry.energy_delta}` : entry.energy_delta}) | H: {entry.entropy_before.toFixed(2)}→{entry.entropy_after.toFixed(2)}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-300">
                    <strong className="text-teal-400">Observed Facts:</strong>{' '}
                    {entry.observed_facts.join(' ')}
                  </div>
                  <div className="text-[11px] text-slate-300">
                    <strong className="text-cyan-400">Inferred Conclusions:</strong>{' '}
                    {entry.inferred_conclusions.join(' ')}
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* Automatic Post-Mission Final Evaluation Report (shown when mission terminates) */}
      {mission.post_mission_evaluation && (
        <div
          className={`border rounded-xl p-4 shadow-xl ${
            mission.post_mission_evaluation.mission_succeeded
              ? 'bg-emerald-950/25 border-emerald-500/50'
              : 'bg-rose-950/25 border-rose-500/50'
          }`}
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-2.5">
            <div className="flex items-center gap-2">
              {mission.post_mission_evaluation.mission_succeeded ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <AlertOctagon className="w-5 h-5 text-rose-400" />
              )}
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-100">
                Automatic Final Mission Report — {mission.post_mission_evaluation.termination_status}
              </h3>
            </div>
            <ShieldCheck className="w-4 h-4 text-slate-400" />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mb-3 font-mono">
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">True EVE Coords</div>
              <div className="text-emerald-300 font-bold">
                ({mission.post_mission_evaluation.true_target_position?.[0]},{' '}
                {mission.post_mission_evaluation.true_target_position?.[1]})
              </div>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">Final MAP Estimate</div>
              <div className="text-cyan-300 font-bold">
                ({mission.post_mission_evaluation.final_estimated_position[0]},{' '}
                {mission.post_mission_evaluation.final_estimated_position[1]}) (
                {(mission.post_mission_evaluation.final_confidence * 100).toFixed(0)}%)
              </div>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">Inferred Status</div>
              <div className="text-amber-300 font-bold truncate">
                {mission.post_mission_evaluation.inferred_eve_status}
              </div>
            </div>
            <div className="bg-slate-950/70 p-2 rounded border border-slate-800">
              <div className="text-[10px] text-slate-400">Actions / Energy</div>
              <div className="text-slate-200 font-bold">
                {mission.post_mission_evaluation.total_actions} steps /{' '}
                {mission.post_mission_evaluation.total_energy_consumed} E
              </div>
            </div>
          </div>

          <div className="space-y-1.5 text-xs text-slate-300">
            {mission.post_mission_evaluation.failure_reason && (
              <p className="text-rose-300 font-medium">
                <strong>Termination Cause:</strong> {mission.post_mission_evaluation.failure_reason}
              </p>
            )}
            <p>
              <strong className="text-slate-100">Evidence Collected:</strong>{' '}
              {mission.post_mission_evaluation.evidence_collected_summary.join(' ')}
            </p>
            <p>
              <strong className="text-slate-100">Remaining Uncertainty:</strong>{' '}
              {mission.post_mission_evaluation.remaining_uncertainty_summary}
            </p>
            <p className="text-teal-300">
              <strong>Recommended Next Action:</strong>{' '}
              {mission.post_mission_evaluation.recommended_next_action}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
