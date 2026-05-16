import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { completeDriveOAuthFromUrl, notifyDriveOAuthComplete } from './driveAuth';
import './index.css';

void completeDriveOAuthFromUrl().then((ok) => {
  if (ok) notifyDriveOAuthComplete();
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
