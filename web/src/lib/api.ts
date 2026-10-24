export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
export const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface JobResult {
  glb_url: string;
  reconstructor: string;
  stats: { bytes: number; vertices?: number; triangles?: number; extents?: number[] };
  timings: Record<string, number>;
  warnings: string[];
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  error: string | null;
  queue_position?: number;
  result?: JobResult;
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {}
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function createJob(file: File): Promise<Job> {
  const form = new FormData();
  form.append("image", file);
  return parse<Job>(await fetch(`${API_URL}/api/v1/jobs`, { method: "POST", body: form }));
}

export async function getJob(id: string): Promise<Job> {
  return parse<Job>(await fetch(`${API_URL}/api/v1/jobs/${id}`, { cache: "no-store" }));
}

export function absoluteUrl(path: string): string {
  return `${API_URL}${path}`;
}

export function checkFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Please choose a PNG or JPG image.";
  if (file.size > MAX_UPLOAD_BYTES) return "Image must be 15 MB or smaller.";
  return null;
}
