export type Coordinate = [number, number];

export type ActionType = 'MOVE' | 'SCAN' | 'WAIT' | 'RECHARGE' | 'RETURN' | 'ABORT';

export type MissionStatus =
  | 'ACTIVE'
  | 'PAUSED'
  | 'SUCCESS'
  | 'FAILED_BATTERY'
  | 'FAILED_TIMEOUT'
  | 'FAILED_UNREACHABLE'
  | 'SAFE_RETURNED'
  | 'ABORTED';

export interface PublicMissionConfig {
  mission_name: string;
  preset_id: string;
  description: string;
  grid_width: number;
  grid_height: number;
  seed: number;
  start_position: Coordinate;
  obstacles: Coordinate[];
  charging_stations: Coordinate[];
  battery_capacity: number;
  move_cost: number;
  scan_cost: number;
  wait_cost: number;
  charge_rate: number;
  sensor_noise_std: number;
  signal_decay_rate: number;
  decoy_count: number;
  decoy_strength: number;
  max_actions: number;
  reveal_target_on_end: boolean;
}

export interface SensorObservation {
  step: number;
  action: ActionType;
  position: Coordinate;
  signal_strength: number | null;
  signal_modulation_hint: string | null;
  adjacent_clear_cells: Coordinate[];
  adjacent_obstacles: Coordinate[];
  cell_inspected: Coordinate;
  target_verified_here: boolean;
  verified_eve_status: string | null;
  is_charging_station: boolean;
  battery_remaining: number;
  energy_delta: number;
  time_remaining_steps: number;
  mission_status: MissionStatus;
  action_succeeded: boolean;
  status_note: string;
}

export interface ChaosEventDetail {
  step: number;
  event_code: string;
  delta_points: number;
  reason: string;
}

export interface ChaosState {
  stars: number;
  raw_score: number;
  status_label: string;
  headline_quote: string;
  recent_reasons: ChaosEventDetail[];
}

export interface RadioMessage {
  step: number;
  event_type: string;
  speaker: string;
  dispatcher_reply: string | null;
  message: string;
}

export interface DecisionExplanation {
  chosen_action: ActionType;
  target_cell: Coordinate;
  objective_mode: string;
  entropy_before: number;
  expected_entropy_reduction: number;
  top_hypothesis_cell: Coordinate;
  top_hypothesis_prob: number;
  inferred_eve_status: string;
  eve_status_distribution: Record<string, number>;
  planned_route: Coordinate[];
  travel_cost_steps: number;
  estimated_energy_required: number;
  nearest_charger_distance: number;
  safety_margin_energy: number;
  risk_assessment: string;
  replanned_due_to_evidence: boolean;
  candidate_scores_summary: Array<Record<string, number | string>>;
  technical_rationale: string;
}

export interface PostMissionEvaluation {
  mission_succeeded: boolean;
  termination_status: MissionStatus;
  failure_reason: string | null;
  true_target_position: Coordinate | null;
  true_decoy_positions: Coordinate[];
  true_eve_status: string | null;
  final_estimated_position: Coordinate;
  final_estimation_error_manhattan: number | null;
  final_confidence: number;
  inferred_eve_status: string;
  inferred_eve_status_confidence: number;
  eve_status_distribution: Record<string, number>;
  remaining_entropy_nats: number;
  remaining_uncertainty_summary: string;
  evidence_collected_summary: string[];
  important_decisions: string[];
  recommended_next_action: string;
  total_actions: number;
  total_energy_consumed: number;
  total_scans: number;
  total_recharges: number;
  cells_explored_count: number;
  coverage_ratio: number;
  closing_commentary: string;
}

export interface PublicMissionState {
  mission_id: string;
  config: PublicMissionConfig;
  status: MissionStatus;
  step_count: number;
  max_actions: number;
  walle_position: Coordinate;
  battery: number;
  battery_capacity: number;
  explored_cells: Coordinate[];
  known_obstacles: Coordinate[];
  scan_history: Array<{ step: number; position: number[]; signal: number; modulation?: string }>;
  belief_map: number[][];
  entropy: number;
  most_likely_cell: Coordinate;
  max_confidence: number;
  inferred_eve_status: string;
  eve_status_belief: Record<string, number>;
  target_verified: boolean;
  current_objective: string;
  planned_route: Coordinate[];
  latest_explanation: DecisionExplanation | null;
  latest_observation: SensorObservation | null;
  latest_commentary: RadioMessage | null;
  commentary_log: RadioMessage[];
  chaos: ChaosState;
  metrics: {
    total_energy_consumed: number;
    total_scans: number;
    total_recharges: number;
    cells_explored: number;
    coverage_pct: number;
  };
  post_mission_evaluation: PostMissionEvaluation | null;
}

export interface StepTraceEntry {
  step: number;
  action: ActionType;
  from_position: Coordinate;
  to_position: Coordinate;
  battery_before: number;
  battery_after: number;
  energy_delta: number;
  mission_status_before: MissionStatus;
  mission_status_after: MissionStatus;
  signal_reading: number | null;
  observed_facts: string[];
  inferred_conclusions: string[];
  uncertainty_metrics: Record<string, number | string>;
  entropy_before: number;
  entropy_after: number;
  most_likely_cell: Coordinate;
  max_confidence: number;
  inferred_eve_status: string;
  eve_status_belief: Record<string, number>;
  replanned: boolean;
  objective_mode: string;
  technical_rationale: string;
  commentary: RadioMessage | null;
  chaos_stars: number;
}

export interface AgentPolicyMetrics {
  policy_name: string;
  noise_std: number;
  trials: number;
  success_rate: number;
  avg_actions_on_success: number;
  avg_actions_overall: number;
  avg_energy_consumed: number;
  avg_scans: number;
  stranded_or_failed_rate: number;
}

export interface ExperimentResponse {
  num_seeds: number;
  noise_levels: number[];
  summary_table: AgentPolicyMetrics[];
  rl_training_curve: Array<{ episode: number; reward: number; steps: number; succeeded: number }> | null;
  interpretation: string;
}
