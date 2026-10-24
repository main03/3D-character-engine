import Studio from "@/components/Studio";
import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1>Photo → 3D Character</h1>
        <p>Upload a photo of a person to generate a textured 3D character you can preview and download as GLB.</p>
      </header>
      <Studio />
    </main>
  );
}
