/**
 * API client for OcuViT backend.
 * Uses VITE_API_URL from environment — never hard-coded.
 */

import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 120_000, // 2 minutes — model inference can be slow on CPU
});

/**
 * Check backend health.
 * @returns {Promise<{status: string, model_loaded: boolean}>}
 */
export async function checkHealth() {
  const { data } = await apiClient.get("/api/health");
  return data;
}

/**
 * Upload an image and get prediction results.
 * @param {File} file — the image file to analyze
 * @returns {Promise<Object>} — prediction response
 */
export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await apiClient.post("/api/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export { API_URL };
