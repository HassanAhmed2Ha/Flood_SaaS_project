const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'https://hassan2007-flood-intelligence-engine.hf.space';
const HF_TOKEN = process.env.HF_TOKEN;

// إعدادات الحماية واستقبال البيانات
app.use(cors());
app.use(express.json());

// Error handler for malformed JSON payloads
app.use((err, req, res, next) => {
    if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
        return res.status(400).json({ error: "Invalid JSON syntax." });
    }
    next();
});

// 1. عرض ملفات الواجهة (Frontend) مباشرة من هذا السيرفر
// نفترض أن مجلد frontend موجود بجانب مجلد backend-node
app.use(express.static(path.join(__dirname, '../frontend/src')));

// 2. نقطة الاتصال (API Gateway) التي تربط الواجهة بمحرك بايثون
app.post('/api/scan', async (req, res) => {
    console.log("\n[Node.js Gateway] 🛰️ Received Scan Request:", req.body);

    try {
        // إرسال الإحداثيات إلى سيرفر بايثون
        console.log("[Node.js Gateway] Forwarding to Python AI Engine...");

        const pythonResponse = await axios.post(`${AI_ENGINE_URL}/api/v1/analyze_flood`, req.body, {
            headers: { 'Authorization': `Bearer ${HF_TOKEN}` },
            timeout: 300000
        });

        console.log("[Node.js Gateway] ✅ Received successful response from AI Engine.");

        // إرسال النتيجة النهائية للواجهة (المتصفح)
        res.json(pythonResponse.data);

    } catch (error) {
        if (error.response) {
            console.error(`[Node.js Gateway] ❌ Upstream error from AI Engine (Status ${error.response.status}):`, error.response.data);
            return res.status(error.response.status).json(error.response.data);
        }
        console.error("[Node.js Gateway] ❌ Error connecting to AI Engine:", error.message);
        res.status(500).json({ error: "Failed to process satellite data via AI Engine." });
    }
});

// تشغيل السيرفر
app.listen(PORT, () => {
    console.log(`=================================================`);
    console.log(`🚀 API Gateway running on http://localhost:${PORT}`);
    console.log(`=================================================`);
});