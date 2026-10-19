import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { Coordinate, PublicMissionState } from '../types/mission';
import { WalleModel } from './WalleModel';

interface Scene3DProps {
  mission: PublicMissionState;
  showHeatmap: boolean;
}

const key = (c: Coordinate | number[]) => `${c[0]},${c[1]}`;

/** Expanding ring emitted at WALL-E's position whenever a SCAN is executed. */
const ScanPulse: React.FC<{ position: [number, number, number]; stepKey: number }> = ({
  position,
  stepKey,
}) => {
  const mesh = useRef<THREE.Mesh>(null);
  const start = useRef<number | null>(null);
  const lastKey = useRef(stepKey);

  useFrame((state) => {
    if (!mesh.current) return;
    if (lastKey.current !== stepKey) {
      lastKey.current = stepKey;
      start.current = null;
    }
    if (start.current === null) start.current = state.clock.elapsedTime;
    const age = state.clock.elapsedTime - start.current;
    const p = Math.min(1, age / 1.1);
    const s = 0.4 + p * 5;
    mesh.current.scale.set(s, s, s);
    (mesh.current.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.7;
  });

  return (
    <mesh ref={mesh} position={[position[0], 0.08, position[2]]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.9, 1, 48]} />
      <meshBasicMaterial color="#22d3ee" transparent opacity={0.7} side={THREE.DoubleSide} />
    </mesh>
  );
};

/** Simple stylised EVE (revealed only in the post-mission evaluation view). */
const EveModel: React.FC<{ position: [number, number, number] }> = ({ position }) => {
  const ref = useRef<THREE.Group>(null);
  useFrame((s) => {
    if (ref.current) ref.current.position.y = 0.55 + Math.sin(s.clock.elapsedTime * 2) * 0.08;
  });
  return (
    <group position={[position[0], 0, position[2]]}>
      <group ref={ref}>
        <mesh scale={[0.9, 1.3, 0.9]} castShadow>
          <sphereGeometry args={[0.28, 24, 24]} />
          <meshStandardMaterial color="#f8fafc" roughness={0.2} metalness={0.2} />
        </mesh>
        <mesh position={[0, 0.12, 0.2]} scale={[1, 0.6, 0.4]}>
          <sphereGeometry args={[0.2, 20, 20]} />
          <meshStandardMaterial color="#020617" roughness={0.1} />
        </mesh>
        {[-0.07, 0.07].map((x) => (
          <mesh key={x} position={[x, 0.13, 0.27]}>
            <sphereGeometry args={[0.03, 10, 10]} />
            <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={3} />
          </mesh>
        ))}
      </group>
      <pointLight position={[0, 0.9, 0]} color="#7dd3fc" intensity={1.2} distance={3} />
    </group>
  );
};

const Floor: React.FC<{ mission: PublicMissionState; showHeatmap: boolean; offset: [number, number] }> = ({
  mission,
  showHeatmap,
  offset,
}) => {
  const { config, belief_map, explored_cells, known_obstacles, max_confidence, scan_history } = mission;
  const exploredSet = useMemo(() => new Set(explored_cells.map(key)), [explored_cells]);
  const obstacleSet = useMemo(() => new Set(known_obstacles.map(key)), [known_obstacles]);
  const chargerSet = useMemo(() => new Set(config.charging_stations.map(key)), [config]);
  const scanSet = useMemo(() => new Set(scan_history.map((s) => key(s.position))), [scan_history]);
  const peak = Math.max(max_confidence, 0.01);

  const tiles: React.ReactNode[] = [];
  for (let x = 0; x < config.grid_width; x++) {
    for (let y = 0; y < config.grid_height; y++) {
      const k = `${x},${y}`;
      const isObstacle = obstacleSet.has(k);
      const prob = belief_map[x]?.[y] ?? 0;
      const rel = showHeatmap && !isObstacle ? Math.min(1, prob / peak) : 0;

      const base = new THREE.Color(exploredSet.has(k) ? '#1b2640' : '#0f172a');
      const heat = new THREE.Color('#06b6d4');
      const color = base.clone().lerp(heat, rel * 0.85);
      const heightBoost = showHeatmap && !isObstacle ? rel * 0.35 : 0;

      tiles.push(
        <group key={k} position={[x + offset[0], 0, y + offset[1]]}>
          {!isObstacle && (
            <mesh position={[0, -0.05 + heightBoost / 2, 0]} receiveShadow>
              <boxGeometry args={[0.94, 0.1 + heightBoost, 0.94]} />
              <meshStandardMaterial
                color={color}
                emissive={heat}
                emissiveIntensity={rel * 0.5}
                roughness={0.7}
                metalness={0.2}
              />
            </mesh>
          )}
          {isObstacle && (
            <mesh position={[0, 0.4, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.92, 0.8 + ((x * 7 + y * 13) % 5) * 0.06, 0.92]} />
              <meshStandardMaterial color="#475569" roughness={0.95} metalness={0.15} />
            </mesh>
          )}
          {chargerSet.has(k) && (
            <group position={[0, 0.05, 0]}>
              <mesh>
                <cylinderGeometry args={[0.33, 0.38, 0.08, 20]} />
                <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.9} />
              </mesh>
              <pointLight color="#34d399" intensity={0.6} distance={2} position={[0, 0.5, 0]} />
            </group>
          )}
          {scanSet.has(k) && !isObstacle && (
            <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.18, 0.24, 24]} />
              <meshBasicMaterial color="#5eead4" />
            </mesh>
          )}
        </group>
      );
    }
  }
  return <>{tiles}</>;
};

const Scene: React.FC<Scene3DProps> = ({ mission, showHeatmap }) => {
  const { config, walle_position, planned_route, most_likely_cell, post_mission_evaluation } = mission;
  const offset: [number, number] = [-(config.grid_width - 1) / 2, -(config.grid_height - 1) / 2];
  const toWorld = (c: Coordinate | number[], y = 0): [number, number, number] => [
    c[0] + offset[0],
    y,
    c[1] + offset[1],
  ];

  const obs = mission.latest_observation;
  const scanning = obs?.action === 'SCAN' && obs.action_succeeded && mission.status === 'ACTIVE';
  const lowBattery = mission.battery <= Math.floor(0.22 * mission.battery_capacity);
  const celebrating = mission.status === 'SUCCESS';

  const target = toWorld(walle_position, 0);
  const mlWorld = toWorld(most_likely_cell, 0);
  const revealTarget = post_mission_evaluation?.true_target_position ?? null;
  const decoys = post_mission_evaluation?.true_decoy_positions ?? [];

  const ml = useRef<THREE.Mesh>(null);
  useFrame((s) => {
    if (ml.current) {
      ml.current.position.y = 1.1 + Math.sin(s.clock.elapsedTime * 3) * 0.08;
      ml.current.rotation.y = s.clock.elapsedTime * 1.5;
    }
  });

  return (
    <>
      <color attach="background" args={['#070b14']} />
      <fog attach="fog" args={['#070b14', 14, 32]} />
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[8, 14, 6]}
        intensity={1.3}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />
      <hemisphereLight args={['#334155', '#0b1120', 0.4]} />

      {/* ground slab */}
      <mesh position={[0, -0.2, 0]} receiveShadow>
        <boxGeometry args={[config.grid_width + 1, 0.2, config.grid_height + 1]} />
        <meshStandardMaterial color="#0b1120" roughness={1} />
      </mesh>

      <Floor mission={mission} showHeatmap={showHeatmap} offset={offset} />

      {/* planned A* route */}
      {planned_route.slice(1).map((c, i) => (
        <mesh key={`r-${i}-${key(c)}`} position={toWorld(c, 0.12)}>
          <sphereGeometry args={[0.08, 12, 12]} />
          <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={2} />
        </mesh>
      ))}

      {/* MAP estimate marker (unverified estimate, not discovery) */}
      {!mission.target_verified && (
        <mesh ref={ml} position={[mlWorld[0], 1.1, mlWorld[2]]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[0.14, 0.3, 4]} />
          <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={1.8} />
        </mesh>
      )}

      {/* Post-mission ground truth reveal */}
      {revealTarget && <EveModel position={toWorld(revealTarget)} />}
      {decoys.map((d) => (
        <group key={key(d)} position={toWorld(d, 0)}>
          <mesh position={[0, 0.3, 0]}>
            <coneGeometry args={[0.2, 0.5, 6]} />
            <meshStandardMaterial color="#f43f5e" emissive="#f43f5e" emissiveIntensity={1.2} />
          </mesh>
          <Text position={[0, 0.8, 0]} fontSize={0.18} color="#fda4af" anchorX="center">
            DECOY
          </Text>
        </group>
      ))}

      <WalleModel target={target} scanning={!!scanning} celebrating={celebrating} lowBattery={lowBattery} />
      {scanning && <ScanPulse position={target} stepKey={mission.step_count} />}

      <OrbitControls
        enablePan={false}
        minDistance={6}
        maxDistance={26}
        maxPolarAngle={Math.PI / 2.15}
        target={[0, 0, 0]}
      />
    </>
  );
};

export const Scene3D: React.FC<Scene3DProps> = ({ mission, showHeatmap }) => {
  return (
    <div className="bg-[#0b1120] border border-slate-800/90 rounded-xl p-2 shadow-xl">
      <div className="flex items-center justify-between px-2 pb-2 text-[11px] font-mono text-slate-400">
        <span>3D Tactical View — drag to orbit, scroll to zoom</span>
        <span className="text-cyan-300">Seed #{mission.config.seed}</span>
      </div>
      <div className="w-full aspect-square max-h-[620px] rounded-lg overflow-hidden">
        <Canvas shadows camera={{ position: [10, 11, 12], fov: 42 }} dpr={[1, 2]}>
          <Scene mission={mission} showHeatmap={showHeatmap} />
        </Canvas>
      </div>
    </div>
  );
};
