import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// --- CONFIGURATION ---
const PORT = 5000;
const VTON_MODE = 'local'; // Options: 'local' (Headless Trials), 'api' (Mirror App - Do not use for server)
const USE_PAID_API = true; // Set to true to use Segmind API (High Quality, Costs Credits)
const TEMP_DIR = './temp_uploads';
const CLOTH_DIR = './datasets/test/cloth'; // Base directory for cloth images

// Setup directories
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (!fs.existsSync(TEMP_DIR)) {
    fs.mkdirSync(TEMP_DIR);
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" })); // Increased limit for detailed images
app.use((req, res, next) => {
    console.log(`[DEBUG] Incoming Request: ${req.method} ${req.path}`);
    next();
});

// Static File Serving
app.use('/garments', express.static(CLOTH_DIR));

// Get Garments Endpoint
app.get('/api/garments', (req, res) => {
    console.log("[Request] Fetching garments list");
    try {
        if (!fs.existsSync(CLOTH_DIR)) {
            return res.json({ images: [] });
        }

        const files = fs.readdirSync(CLOTH_DIR);
        // Limit to 200 to prevent payload issues, randomizing to show variety
        const images = files.filter(f => /\.(jpg|jpeg|png)$/i.test(f));
        const limitedImages = images.sort(() => 0.5 - Math.random()).slice(0, 200);

        return res.json({ images: limitedImages });
    } catch (e) {
        console.error("Error listing garments:", e);
        return res.status(500).json({ error: "Failed to list garments" });
    }
});

// Recommendation Endpoint (Mock)
app.post('/recommend', (req, res) => {
    console.log("[Request] Received Recommendation request");

    // Mock Logic: Return 4 random images from the cloth directory
    try {
        if (!fs.existsSync(CLOTH_DIR)) {
            return res.json({ recommendations: [] });
        }
        const files = fs.readdirSync(CLOTH_DIR);
        const images = files.filter(f => /\.(jpg|jpeg|png)$/i.test(f));

        // Pick 4 random
        const recommendations = [];
        for (let i = 0; i < 4; i++) {
            if (images.length > 0) {
                const randomImg = images[Math.floor(Math.random() * images.length)];
                recommendations.push({
                    category: 'Recommended',
                    image_id: randomImg
                });
            }
        }
        return res.json({ recommendations });
    } catch (e) {
        console.error("Rec Error:", e);
        return res.json({ recommendations: [] });
    }
});


// VTON Endpoint
app.post('/api/tryon', (req, res) => {
    console.log(`[Request] Received VTON request (Mode: ${VTON_MODE})`);

    const { user_image, garment_image, height, category } = req.body;

    if (!user_image || !garment_image) {
        return res.status(400).json({ error: "Missing user_image or garment_image" });
    }

    try {
        // 1. Save Input Images
        const timestamp = Date.now();
        const inputPath = path.join(TEMP_DIR, `input_${timestamp}.jpg`);
        // Handle both base64 (from canvas) and URLs
        const base64Data = user_image.replace(/^data:image\/\w+;base64,/, "");
        fs.writeFileSync(inputPath, Buffer.from(base64Data, 'base64'));

        console.log(`[IO] Saved input image to: ${inputPath}`);

        // 2. Prepare Python Command
        // We need to resolve the local path for the garment if it's a URL
        let garmentFilename = path.basename(garment_image);
        let garmentPath = path.join(CLOTH_DIR, garmentFilename);

        // Sanity check if garment exists locally
        if (!fs.existsSync(garmentPath)) {
            console.warn(`[Warning] Garment file not found locally: ${garmentPath}`);
        }

        // Map frontend categories to Python engine accepted choices
        let engineCategory = category || 'Upper-body';
        if (engineCategory === 'T-SHIRTS') engineCategory = 'Upper-body';

        // 3. Spawn Python Process
        // Command: python run_trial.py --image <path> --cloth <path> --height <val> --category <val> --provider segmind
        const pythonArgs = [
            'run_trial.py',
            '--image', inputPath,
            '--cloth', garmentPath,
            '--height', height || '170',
            '--category', engineCategory,
            '--provider', 'segmind'
        ];

        console.log("[Process] Spawning Python engine...");
        const pythonProcess = spawn('python', pythonArgs);

        let dataString = '';
        let errorString = '';

        pythonProcess.stdout.on('data', (data) => {
            dataString += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorString += data.toString();
            console.error(`[Python Error]: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`[Process] Python exited with code ${code}`);

            // Looking for: __MIRAI_OUTPUT__ {"success": true, "result_path": "..."}
            const match = dataString.match(/__MIRAI_OUTPUT__(.+)/);
            if (match && match[1]) {
                try {
                    const outputData = JSON.parse(match[1]);
                    if (outputData.success && outputData.result_path) {
                        const resultPath = outputData.result_path.trim();
                        console.log(`[Success] Found result at: ${resultPath}`);

                        if (fs.existsSync(resultPath)) {
                            const resultBuffer = fs.readFileSync(resultPath);
                            const resultBase64 = `data:image/jpeg;base64,${resultBuffer.toString('base64')}`;
                            return res.json({ result_image: resultBase64 });
                        }
                    }
                } catch (e) {
                    console.error("[Error] Failed to parse Python JSON output:", e);
                }
            }

            console.error("[Error] No valid result found in python output");
            // Debugging: Print last few lines of output
            console.error("Partial Output:", dataString.slice(-500));
            return res.status(500).json({ error: "No result generated", details: errorString });
        });

    } catch (e) {
        console.error("Server Error:", e);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

app.listen(PORT, () => {
    console.log(`MIRAI VTON Server is listening on port ${PORT}...`);
    console.log(`Mode: ${VTON_MODE.toUpperCase()} | Provider: ${USE_PAID_API ? 'SEGMIND (PAID)' : 'GRADIO (FREE)'}`);
});
