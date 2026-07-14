require('dotenv').config({ path: '.env.local' });

const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3000;
const BACKEND_TARGET = process.env.backend || 'http://localhost:8000';

// Proxy API calls to FastAPI backend
app.use('/api', createProxyMiddleware({
  target: BACKEND_TARGET,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
}));
 
// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
 
app.listen(PORT, () => {
  console.log(`Frontend on http://localhost:${PORT}`);
  console.log(`Proxying /api -> ${BACKEND_TARGET}`);
});
