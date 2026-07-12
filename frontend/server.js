const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
 
const app = express();
const PORT = 3000;
 
// Proxy API calls to FastAPI backend
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
}));
 
// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
 
app.listen(PORT, () => {
  console.log(`Frontend on http://localhost:${PORT}`);
});
