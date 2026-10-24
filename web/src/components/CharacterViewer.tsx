"use client";

import { Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, ContactShadows, Html, OrbitControls, useGLTF, useProgress } from "@react-three/drei";

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  useEffect(() => () => useGLTF.clear(url), [url]);
  return <primitive object={scene} />;
}

function Loader() {
  const { progress } = useProgress();
  return <Html center>{Math.round(progress)}%</Html>;
}

export default function CharacterViewer({ url }: { url: string }) {
  return (
    <Canvas camera={{ position: [0, 0.8, 2.4], fov: 35 }} dpr={[1, 2]}>
      <color attach="background" args={["#1b1d22"]} />
      <hemisphereLight args={["#ffffff", "#444444", 1.6]} />
      <directionalLight position={[2, 3, 3]} intensity={1.6} />
      <directionalLight position={[-2, 1, -3]} intensity={0.6} />
      <Suspense fallback={<Loader />}>
        <Bounds fit clip observe margin={1.2}>
          <Model url={url} />
        </Bounds>
        <ContactShadows position={[0, 0, 0]} opacity={0.5} scale={3} blur={2} far={1.5} />
      </Suspense>
      <OrbitControls makeDefault enableDamping minDistance={0.3} maxDistance={8} />
    </Canvas>
  );
}
