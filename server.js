// server.js
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');

/* ===========================
   🔹 NEW: Groq / LLaMA Setup
   =========================== */
const Groq = require('groq-sdk');
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

// 🔹 In-memory AI cache (simple & fast)
const aiInsightCache = new Map();

/* ===========================
   🔹 NEW: AI Helper Functions
   =========================== */
function buildAIInsightPayload(compareJson) {
  const regions = compareJson.regions || [];
  let red = 0, yellow = 0, green = 0;

  regions.forEach(r => {
    const sev = (r.sev || "").toLowerCase();
    if (sev === "red") red++;
    else if (sev === "yellow") yellow++;
    else green++;
  });

  return {
    total_regions: regions.length,
    severity_breakdown: { red, yellow, green },
    ssim_score: compareJson.ssim_score
  };
}

function buildInsightTable(compareJson) {
  const regions = compareJson.regions || [];

  let red = 0, yellow = 0, green = 0;
  regions.forEach(r => {
    const sev = (r.sev || "").toLowerCase();
    if (sev === "red") red++;
    else if (sev === "yellow") yellow++;
    else green++;
  });

  return [
    {
      metric: "Total Changed Regions",
      value: regions.length,
      interpretation: "Number of areas where visual differences were detected"
    },
    {
      metric: "High Severity Changes",
      value: red,
      interpretation: "Critical or clearly visible changes"
    },
    {
      metric: "Medium Severity Changes",
      value: yellow,
      interpretation: "Noticeable but non-critical changes"
    },
    {
      metric: "Low Severity Changes",
      value: green,
      interpretation: "Minor or subtle differences"
    },
    {
      metric: "Overall Similarity (SSIM)",
      value: compareJson.ssim_score,
      interpretation: "Higher value means images are more similar"
    }
  ];
}

function buildCacheKey(compareJson) {
  const regions = compareJson.regions || [];
  const ssim = compareJson.ssim_score || 0;

  return `regions:${regions.length}|ssim:${ssim.toFixed(3)}`;
}


function buildPrompt(payload) {
  return `
You are an AI assistant helping users analyze visual changes across images in inspection and monitoring scenarios such as motorsports, manufacturing, or compliance checks.

Image comparison results:
- Structural Similarity (SSIM): ${payload.ssim_score}
- Total changed regions: ${payload.total_regions}
- High severity changes: ${payload.severity_breakdown.red}
- Medium severity changes: ${payload.severity_breakdown.yellow}
- Low severity changes: ${payload.severity_breakdown.green}

Task:
Write a concise, professional explanation of what changed between the images.
Guidelines:
- Prioritize high and medium severity changes if present
- Explain whether changes indicate damage, movement, or minor variation
- If changes are mostly low severity, clearly state that no critical issues were detected
- Keep the explanation understandable for non-technical users
- Do not mention algorithms or implementation details
`;
}


async function generateAIInsight(prompt) {
  const completion = await groq.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: "You explain computer vision results clearly." },
      { role: "user", content: prompt }
    ],
    temperature: 0.3,
    max_tokens: 200
  });

  return completion.choices[0].message.content;
}

/* ===========================
   🔹 EXISTING SERVER CODE
   =========================== */

const app = express();
app.use(cors());
app.use(express.json());

try {
  const authRouter = require('./routes/auth');
  if (authRouter) app.use('/api/auth', authRouter);
} catch (e) {}

const tmpDir = path.join(__dirname, 'tmp');
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

const upload = multer({ dest: tmpDir });

/* ===========================
   🔹 /api/compare (RESTORED HEATMAP)
   =========================== */
app.post('/api/compare', upload.fields([
  { name: 'beforeFile', maxCount: 1 },
  { name: 'afterFile', maxCount: 1 }
]), async (req, res) => {
  try {
    if (!req.files || !req.files.beforeFile || !req.files.afterFile) {
      return res.status(400).json({ success: false, error: 'Please send both beforeFile and afterFile' });
    }

    const beforePath = req.files.beforeFile[0].path;
    const afterPath = req.files.afterFile[0].path;

    // Prefer an explicit PYTHON_PATH if provided; otherwise pick a sensible default
    // 'python3' is common on Linux, while 'python' may be the default on Windows.
    const pythonExec = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'python' : 'python3');
    const py = spawn(pythonExec, [path.join(__dirname, 'compare.py'), beforePath, afterPath]);

    let stdout = '';
    let stderr = '';
    py.stdout.on('data', d => stdout += d.toString());
    py.stderr.on('data', d => stderr += d.toString());

    const exitCode = await new Promise(r => py.on('close', r));
    if (exitCode !== 0) {
      return res.status(500).json({ success: false, error: stderr });
    }

    let parsedPy;
    try {
      parsedPy = JSON.parse(stdout);
    } catch {
      return res.status(500).json({ success: false, error: 'Invalid compare output' });
    }

    const compResult = parsedPy.compareJson || parsedPy;

    /* 🔥 RESTORED HEATMAP EXTRACTION */
    const heatmapFileLocal = parsedPy.heatmap_file || parsedPy.heatmapFile || null;

    let uploadRes = null;
    if (
      heatmapFileLocal &&
      process.env.CLOUDINARY_CLOUD_NAME &&
      process.env.CLOUDINARY_API_KEY &&
      process.env.CLOUDINARY_API_SECRET
    ) {
      try {
        const cloudinary = require('cloudinary').v2;
        cloudinary.config({
          cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
          api_key: process.env.CLOUDINARY_API_KEY,
          api_secret: process.env.CLOUDINARY_API_SECRET,
          secure: true
        });

        uploadRes = await cloudinary.uploader.upload(
          heatmapFileLocal,
          { folder: 'visuview/heatmaps' }
        );
      } catch (err) {
        console.warn('Cloudinary upload failed:', err.message);
      }
    }

    try { fs.unlinkSync(beforePath); fs.unlinkSync(afterPath); } catch {}

    const safeCompareJson = {
      regions: compResult.regions || [],
      meta: compResult.meta || {},
      summary: compResult.summary || null,
      ssim_score: compResult.ssim_score ?? parsedPy.score ?? null
    };

    return res.json({
      success: true,
      compareJson: safeCompareJson,
      heatmapPublicId: uploadRes?.public_id || null,
      heatmapUrl: uploadRes?.secure_url || null
    });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ success: false, error: String(err) });
  }
});

/* ===========================
   🔹 NEW: AI INSIGHT ENDPOINT
   =========================== */
app.post('/api/ai-insight', async (req, res) => {
  try {
    const { compareJson } = req.body;
    if (!compareJson) {
      return res.status(400).json({ success: false, error: 'Missing compareJson' });
    }

    const cacheKey = buildCacheKey(compareJson);

// 1️⃣ Check cache first
if (aiInsightCache.has(cacheKey)) {
  return res.json({
    success: true,
    ...aiInsightCache.get(cacheKey),
    cached: true
  });
}

// 2️⃣ Not cached → generate AI insight
const payload = buildAIInsightPayload(compareJson);
const prompt = buildPrompt(payload);
const insight = await generateAIInsight(prompt);
const table = buildInsightTable(compareJson);

// 3️⃣ Save to cache
aiInsightCache.set(cacheKey, {
  insight,
  table
});

// 4️⃣ Return response
return res.json({
  success: true,
  insight,
  table,
  cached: false
});

  } catch (err) {
    console.error('AI Insight error:', err);
    res.status(500).json({ success: false, error: 'AI generation failed' });
  }
});

// health
app.get('/api/health', (req, res) => res.json({ ok: true }));

app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
