// import { defineConfig } from "vite";
// import react from "@vitejs/plugin-react";

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 3000, // Port untuk development server
//     proxy: {
//       // Proxy API requests ke backend Flask
//       "/api": {
//         target: "http://localhost:5000", // Alamat backend Flask
//         changeOrigin: true,
//         secure: false,
//         rewrite: (path) => path.replace(/^\/api/, ""),
//       },
//       "/login/google": {
//         target: "http://localhost:5000",
//         changeOrigin: true,
//         secure: false,
//       },
//       "/login/google/authorized": {
//         target: "http://localhost:5000",
//         changeOrigin: true,
//         secure: false,
//       },
//     },
//     allowedHosts: ["obliging-flamingo-deadly.ngrok-free.app"], // ⬅️ Tambahkan ini
//   },
//   build: {
//     outDir: "dist",
//     assetsDir: "assets",
//     sourcemap: true,
//   },
//   resolve: {
//     alias: {
//       "@": "/src",
//     },
//   },
//   optimizeDeps: {
//     include: ["react", "react-dom"],
//   },
// });

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
});
