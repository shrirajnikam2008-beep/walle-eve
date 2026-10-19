import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface WalleModelProps {
  /** World-space target position (grid cell center). The model glides toward it. */
  target: [number, number, number];
  /** True while a SCAN action was just executed (antenna/eyes flare). */
  scanning: boolean;
  /** True when the mission is finished successfully (celebration wobble). */
  celebrating: boolean;
  /** True when battery is critically low (eyes turn red & droop). */
  lowBattery: boolean;
}

const BODY = '#e2a526';
const BODY_DARK = '#b57f12';
const RUST = '#8a5a14';
const TRACK = '#2a2a2e';
const METAL = '#6b6f78';

/**
 * Procedural 3D WALL-E built from primitives (no external asset files needed).
 * Pure presentation: it never influences planning, beliefs, or metrics.
 */
export const WalleModel: React.FC<WalleModelProps> = ({
  target,
  scanning,
  celebrating,
  lowBattery,
}) => {
  const root = useRef<THREE.Group>(null);
  const head = useRef<THREE.Group>(null);
  const leftEye = useRef<THREE.Group>(null);
  const rightEye = useRef<THREE.Group>(null);
  const leftArm = useRef<THREE.Group>(null);
  const rightArm = useRef<THREE.Group>(null);
  const treadL = useRef<THREE.Mesh>(null);
  const treadR = useRef<THREE.Mesh>(null);
  const lensL = useRef<THREE.MeshStandardMaterial>(null);
  const lensR = useRef<THREE.MeshStandardMaterial>(null);
  const heading = useRef(0);
  const initialized = useRef(false);

  useFrame((state, delta) => {
    const g = root.current;
    if (!g) return;
    const t = state.clock.elapsedTime;

    if (!initialized.current) {
      g.position.set(target[0], target[1], target[2]);
      initialized.current = true;
    }

    // Glide toward target cell
    const dx = target[0] - g.position.x;
    const dz = target[2] - g.position.z;
    const dist = Math.hypot(dx, dz);
    const moving = dist > 0.02;
    const k = 1 - Math.exp(-7 * delta);
    g.position.x += dx * k;
    g.position.z += dz * k;
    g.position.y = target[1];

    // Turn toward travel direction (shortest angle)
    if (moving) {
      const desired = Math.atan2(dx, dz);
      let diff = desired - heading.current;
      diff = Math.atan2(Math.sin(diff), Math.cos(diff));
      heading.current += diff * (1 - Math.exp(-10 * delta));
    }
    g.rotation.y = heading.current;

    // Idle bob / drive rumble / celebration
    const bob = moving ? Math.sin(t * 24) * 0.012 : Math.sin(t * 2) * 0.006;
    g.position.y += bob;
    if (celebrating) {
      g.position.y += Math.abs(Math.sin(t * 6)) * 0.12;
      g.rotation.y += Math.sin(t * 5) * 0.25;
    }

    // Track tread scroll illusion
    const spin = moving ? t * 14 : 0;
    if (treadL.current) treadL.current.rotation.x = spin;
    if (treadR.current) treadR.current.rotation.x = spin;

    // Head look-around, scan nod
    if (head.current) {
      head.current.rotation.y = Math.sin(t * 1.3) * 0.35;
      head.current.rotation.x = scanning ? -0.35 + Math.sin(t * 18) * 0.08 : lowBattery ? 0.45 : Math.sin(t * 1.7) * 0.06;
    }

    // Eyes: independent wobble, droop when battery low
    const droop = lowBattery ? 0.5 : 0;
    if (leftEye.current) leftEye.current.rotation.z = Math.sin(t * 2.1) * 0.12 + droop;
    if (rightEye.current) rightEye.current.rotation.z = Math.sin(t * 2.4 + 1) * 0.12 - droop;

    // Arms swing while moving
    const swing = moving ? Math.sin(t * 12) * 0.35 : Math.sin(t * 1.5) * 0.05;
    if (leftArm.current) leftArm.current.rotation.x = swing;
    if (rightArm.current) rightArm.current.rotation.x = -swing;

    // Lens glow: teal normally, white flare when scanning, red when low
    const col = lowBattery ? '#ef4444' : scanning ? '#ffffff' : '#22d3ee';
    const intensity = scanning ? 3 + Math.sin(t * 20) : lowBattery ? 0.8 + Math.sin(t * 6) * 0.5 : 1.4;
    [lensL.current, lensR.current].forEach((m) => {
      if (!m) return;
      m.emissive.set(col);
      m.emissiveIntensity = intensity;
    });
  });

  return (
    <group ref={root} scale={0.5}>
      {/* ---- Tracks ---- */}
      {[-0.62, 0.62].map((x, i) => (
        <group key={i} position={[x, 0.28, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.32, 0.5, 1.5]} />
            <meshStandardMaterial color={TRACK} roughness={0.9} />
          </mesh>
          {/* tread ribs */}
          <mesh ref={i === 0 ? treadL : treadR}>
            <cylinderGeometry args={[0.26, 0.26, 0.34, 8]} />
            <meshStandardMaterial color="#3b3b42" roughness={0.8} />
          </mesh>
          {[-0.55, -0.18, 0.18, 0.55].map((z, j) => (
            <mesh key={j} position={[i === 0 ? -0.17 : 0.17, 0, z]}>
              <boxGeometry args={[0.04, 0.52, 0.1]} />
              <meshStandardMaterial color="#1a1a1d" />
            </mesh>
          ))}
        </group>
      ))}

      {/* ---- Body (compactor cube) ---- */}
      <mesh position={[0, 0.98, 0]} castShadow>
        <boxGeometry args={[1.05, 1.0, 1.15]} />
        <meshStandardMaterial color={BODY} roughness={0.55} metalness={0.25} />
      </mesh>
      {/* body panel lines / rust bands */}
      <mesh position={[0, 0.7, 0]}>
        <boxGeometry args={[1.08, 0.08, 1.18]} />
        <meshStandardMaterial color={RUST} roughness={0.9} />
      </mesh>
      <mesh position={[0, 1.28, 0]}>
        <boxGeometry args={[1.08, 0.05, 1.18]} />
        <meshStandardMaterial color={BODY_DARK} roughness={0.8} />
      </mesh>
      {/* chest solar panel + status light */}
      <mesh position={[0, 1.0, 0.585]}>
        <boxGeometry args={[0.72, 0.5, 0.03]} />
        <meshStandardMaterial color="#1e293b" metalness={0.6} roughness={0.35} />
      </mesh>
      <mesh position={[0.26, 1.18, 0.605]}>
        <circleGeometry args={[0.05, 16]} />
        <meshStandardMaterial
          color={lowBattery ? '#ef4444' : '#22c55e'}
          emissive={lowBattery ? '#ef4444' : '#22c55e'}
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* back vent */}
      <mesh position={[0, 1.0, -0.585]}>
        <boxGeometry args={[0.6, 0.45, 0.03]} />
        <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.5} />
      </mesh>

      {/* ---- Neck + Head ---- */}
      <mesh position={[0, 1.6, 0.1]} castShadow>
        <cylinderGeometry args={[0.1, 0.12, 0.45, 12]} />
        <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.4} />
      </mesh>
      <group ref={head} position={[0, 1.88, 0.1]}>
        {/* eye bridge */}
        <mesh>
          <boxGeometry args={[0.7, 0.18, 0.22]} />
          <meshStandardMaterial color={BODY_DARK} metalness={0.4} roughness={0.5} />
        </mesh>
        {/* binocular eyes */}
        {[-0.3, 0.3].map((x, i) => (
          <group key={i} ref={i === 0 ? leftEye : rightEye} position={[x, 0.08, 0.1]}>
            <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
              <cylinderGeometry args={[0.2, 0.2, 0.42, 20]} />
              <meshStandardMaterial color="#cfd2d8" metalness={0.8} roughness={0.3} />
            </mesh>
            {/* lens */}
            <mesh position={[0, 0, 0.22]} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[0.15, 0.15, 0.04, 20]} />
              <meshStandardMaterial
                ref={i === 0 ? lensL : lensR}
                color="#0b1120"
                emissive="#22d3ee"
                emissiveIntensity={1.4}
              />
            </mesh>
            {/* eyebrow/brow ring */}
            <mesh position={[0, 0, 0.12]} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[0.2, 0.025, 8, 20]} />
              <meshStandardMaterial color={BODY_DARK} />
            </mesh>
          </group>
        ))}
        {/* antenna */}
        <mesh position={[0, 0.25, -0.05]}>
          <cylinderGeometry args={[0.012, 0.012, 0.3, 6]} />
          <meshStandardMaterial color={METAL} />
        </mesh>
        <mesh position={[0, 0.42, -0.05]}>
          <sphereGeometry args={[0.04, 10, 10]} />
          <meshStandardMaterial
            color="#f43f5e"
            emissive="#f43f5e"
            emissiveIntensity={scanning ? 3 : 0.6}
          />
        </mesh>
      </group>

      {/* ---- Arms ---- */}
      {[-1, 1].map((side, i) => (
        <group key={i} ref={i === 0 ? leftArm : rightArm} position={[side * 0.62, 1.2, 0.2]}>
          <mesh position={[0, -0.28, 0]} castShadow>
            <boxGeometry args={[0.1, 0.55, 0.1]} />
            <meshStandardMaterial color={METAL} metalness={0.6} roughness={0.5} />
          </mesh>
          {/* claw hand */}
          <mesh position={[0, -0.6, 0.05]}>
            <boxGeometry args={[0.16, 0.12, 0.2]} />
            <meshStandardMaterial color={BODY_DARK} roughness={0.7} />
          </mesh>
        </group>
      ))}
    </group>
  );
};
