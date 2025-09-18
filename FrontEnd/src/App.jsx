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
      {/* halaman awal */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/app" element={<MainLayout />}>
        {/* Rute default di dalam /app adalah chat */}
        <Route index element={<Navigate to="chat" replace />} />
        <Route path="chat" element={<ChatInterface />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="help" element={<HelpPage />} />
      </Route>
      
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;