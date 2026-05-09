import { Navigate, Route, Routes } from 'react-router-dom';
import DashboardLayout from '../components/layout/DashboardLayout';
import CrowdMonitoringPage from '../pages/crowd-monitoring';
import WaterQualityPage from '../pages/water-quality';
import DrowningDetectionPage from '../pages/drowning-detection';
import GarbageDetectionPage from '../pages/garbage-detection';

export default function AppRouter() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route index element={<Navigate to="/crowd-monitoring" replace />} />
        <Route path="/crowd-monitoring" element={<CrowdMonitoringPage />} />
        <Route path="/water-quality" element={<WaterQualityPage />} />
        <Route path="/drowning-detection" element={<DrowningDetectionPage />} />
        <Route path="/garbage-detection" element={<GarbageDetectionPage />} />
      </Route>
    </Routes>
  );
}
