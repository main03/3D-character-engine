"use client";

import { Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, ContactShadows, OrbitControls, useGLTF, useProgress } from "@react-three/drei";
import { NeutralToneMapping } from "three";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  useEffect(() => () => useGLTF.clear(url), [url]);
  return <primitive object={scene} />;
}

// Rendered as plain DOM outside the Canvas: drei's <Html> inside a Suspense fallback
// races React 19's unmounting ("removeChild" / "synchronously unmount a root" errors).
function LoadingOverlay() {
  const { active, progress } = useProgress();
  if (!active) return null;
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#c9ced8",
        pointerEvents: "none",
      }}
    >
      Loading model… {Math.round(progress)}%
    </div>
  );
}

export default function CharacterViewer({ url }: { url: string }) {
  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <Canvas
        camera={{ position: [0, 0.8, 2.4], fov: 35 }}
        dpr={[1, 2]}
        gl={{ toneMapping: NeutralToneMapping }}
      >
        <color attach="background" args={["#1b1d22"]} />
        <hemisphereLight args={["#ffffff", "#5a5a5a", 1.1]} />
        <directionalLight position={[1.5, 2, 3]} intensity={1.1} />
        <directionalLight position={[-2, 1, -3]} intensity={0.4} />
        <Suspense fallback={null}>
          <Bounds fit clip observe margin={1.2}>
            <Model url={url} />
          </Bounds>
          <ContactShadows position={[0, 0, 0]} opacity={0.5} scale={3} blur={2} far={1.5} />
        </Suspense>
        <OrbitControls makeDefault enableDamping minDistance={0.3} maxDistance={8} />
      </Canvas>
      <LoadingOverlay />
    </div>
  );
}
