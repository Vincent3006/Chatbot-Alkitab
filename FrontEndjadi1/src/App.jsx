import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import MainLayout from './components/MainLayout';
import ChatInterface from './components/ChatInterface';
import AboutPage from './components/AboutPage';
import HelpPage from './components/HelpPage';
import './App.css';

function App() {
  return (
    <Routes>
      {/* Rute untuk halaman awal */}
      <Route path="/" element={<LandingPage />} />
      
      {/* Rute untuk layout utama aplikasi setelah login */}
      <Route path="/app" element={<MainLayout />}>
        {/* Rute default di dalam /app adalah chat */}
        <Route index element={<Navigate to="chat" replace />} />
        <Route path="chat" element={<ChatInterface />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="help" element={<HelpPage />} />
      </Route>
      
      {/* Jika pengguna mengakses rute lain, arahkan ke halaman awal */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;