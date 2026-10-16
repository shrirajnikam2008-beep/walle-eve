import {
  ExperimentResponse,
  PublicMissionConfig,
  PublicMissionState,
  StepTraceEntry,
} from '../types/mission';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) detail = errJson.detail;
    } catch {
      // ignore parse failure
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const missionApi = {
  checkHealth: () => request<{ status: string; presets_available: number }>('/health'),

  listPresets: () => request<PublicMissionConfig[]>('/presets'),

  createMission: (presetId: string, humorEnabled: boolean) =>
    request<PublicMissionState>('/missions', {
      method: 'POST',
      body: JSON.stringify({ preset_id: presetId, humor_enabled: humorEnabled }),
    }),

  getMission: (missionId: string) => request<PublicMissionState>(`/missions/${missionId}`),

  stepMission: (missionId: string, humorEnabled: boolean) =>
    request<PublicMissionState>(
      `/missions/${missionId}/step?humor_enabled=${humorEnabled}`,
      { method: 'POST' }
    ),

  runMissionFast: (missionId: string, humorEnabled: boolean, maxSteps = 120) =>
    request<PublicMissionState>(`/missions/${missionId}/run`, {
      method: 'POST',
      body: JSON.stringify({ max_steps: maxSteps, humor_enabled: humorEnabled }),
    }),

  pauseMission: (missionId: string) =>
    request<PublicMissionState>(`/missions/${missionId}/pause`, { method: 'POST' }),

  resetMission: (missionId: string, humorEnabled: boolean) =>
    request<PublicMissionState>(
      `/missions/${missionId}/reset?humor_enabled=${humorEnabled}`,
      { method: 'POST' }
    ),

  getTrace: (missionId: string) =>
    request<StepTraceEntry[]>(`/missions/${missionId}/trace`),

  runExperiments: (numSeeds = 25, includeRl = true) =>
    request<ExperimentResponse>('/experiments/run', {
      method: 'POST',
      body: JSON.stringify({
        num_seeds: numSeeds,
        base_seed: 100,
        noise_levels: [0.06, 0.10, 0.16],
        include_rl_agent: includeRl,
      }),
    }),
};
