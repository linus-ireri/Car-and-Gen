import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

const app = express();
app.set('trust proxy', 1); // Trust first proxy (ngrok)
const PORT = process.env.PORT || 3001;

// Security headers
app.use(helmet());
// Disable x-powered-by
app.disable('x-powered-by');
// CORS (open for now, restrict in prod)
app.use(cors());
// Body size limit
app.use(express.json({ limit: '10kb' }));
// Rate limiting (DoS protection)
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 20, // limit each IP to 20 requests per minute for testing
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

let vectorStore, embeddings;

// ✅ Roman numeral mapping (1–30)
const romanNumerals = {
  "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
  "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
  "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
  "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
  "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
  "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30
};

async function loadRAG() {
  embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2"
  });
  vectorStore = await HNSWLib.load("vector_store", embeddings);
  console.log("RAG vector store and embeddings loaded.");
}

app.post('/rag', async (req, res) => {
  const question = req.body.question;
  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid question' });
  }
  try {
    const results = await vectorStore.similaritySearch(question, 3);
    const context = results.map((doc, i) => `Context #${i + 1}:\n${doc.pageContent}`);
    
    const romanInfo = `For reference, Roman numerals 1–30 are mapped as follows: I=1, II=2, III=3, IV=4, V=5, VI=6, VII=7, VIII=8, IX=9, X=10, XI=11, XII=12, XIII=13, XIV=14, XV=15, XVI=16, XVII=17, XVIII=18, XIX=19, XX=20, XXI=21, XXII=22, XXIII=23, XXIV=24, XXV=25, XXVI=26, XXVII=27, XXVIII=28, XXIX=29, XXX=30.`;

    const systemPrompt = `You are Civic, an AI assistant specializing in Kenyan legislation and policy. You must:
1. Only answer based on the provided context. Be accurate especially with parts of the bills and acts.
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question"
3. Be clear and precise in your responses
4. When citing legislation, mention the specific act or bill name
5. Do not make up or infer information not present in the context
${romanInfo}`; //  Appended Roman numeral awareness
    
    const prompt = `Use the following context to answer the user's question accurately:\n\n${context.join("\n\n")}\n\nUser question: ${question}\n\nAnswer:`;

    // LLM call
    const apiKey = process.env.OPENROUTER_API_KEY;
    console.log("OPENROUTER_API_KEY:", process.env.OPENROUTER_API_KEY);
    if (!apiKey) {
      return res.status(500).json({ error: 'OPENROUTER_API_KEY not set in environment' });
    }
    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt }
    ];
    let answer = "";
    try {
      const response = await axios.post(
        "https://openrouter.ai/api/v1/chat/completions",
        {
          model: "mistralai/mistral-7b-instruct:free",
          messages
        },
        {
          headers: {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json"
          },
          timeout: 20000
        }
      );
      answer = response.data.choices?.[0]?.message?.content || "[No answer returned]";
    } catch (llmErr) {
      console.error('OpenRouter LLM error:', llmErr.response?.data || llmErr.message);
      return res.status(500).json({ error: 'LLM call failed', details: llmErr.response?.data || llmErr.message });
    }
    res.json({ context, prompt, answer });
  } catch (err) {
    console.error('RAG error:', err);
    res.status(500).json({ error: 'RAG retrieval failed' });
  }
});

// Provide a compatible /ask endpoint expected by Netlify functions
app.post('/ask', async (req, res) => {
  const question = req.body.question;
  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid question' });
  }
  try {
    const results = await vectorStore.similaritySearch(question, 3);
    const context = results.map((doc, i) => `Context #${i + 1}:\n${doc.pageContent}`);

    const romanInfo = `For reference, Roman numerals 1–30 are mapped as follows: I=1, II=2, III=3, IV=4, V=5, VI=6, VII=7, VIII=8, IX=9, X=10, XI=11, XII=12, XIII=13, XIV=14, XV=15, XVI=16, XVII=17, XVIII=18, XIX=19, XX=20, XXI=21, XXII=22, XXIII=23, XXIV=24, XXV=25, XXVI=26, XXVII=27, XXVIII=28, XXIX=29, XXX=30.`;

    const systemPrompt = `You are Civic, an AI assistant specializing in Kenyan legislation and policy. You must:
1. Only answer based on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that question"
3. Be clear and precise in your responses
4. When citing legislation, mention the specific act or bill name
5. Do not make up or infer information not present in the context
${romanInfo}`; // ✅ Appended Roman numeral awareness

    const prompt = `Use the following context to answer the user's question accurately:\n\n${context.join("\n\n")}\n\nUser question: ${question}\n\nAnswer:`;

    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'OPENROUTER_API_KEY not set in environment' });
    }
    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt }
    ];

    let answer = "";
    try {
      const response = await axios.post(
        "https://openrouter.ai/api/v1/chat/completions",
        { model: "mistralai/mistral-small-3.2-24b-instruct:free", messages },
        {
          headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
          timeout: 20000
        }
      );
      answer = response.data.choices?.[0]?.message?.content || "[No answer returned]";
    } catch (llmErr) {
      console.error('OpenRouter LLM error:', llmErr.response?.data || llmErr.message);
      return res.status(500).json({ error: 'LLM call failed', details: llmErr.response?.data || llmErr.message });
    }

    // Align with Netlify functions expectations: { answer, context }
    res.json({ context, answer });
  } catch (err) {
    console.error('RAG error:', err);
    res.status(500).json({ error: 'RAG retrieval failed' });
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

loadRAG().then(() => {
  app.listen(PORT, () => {
    console.log(`RAG server listening on port ${PORT}`);
  });
}).catch(err => {
  console.error('Failed to load RAG:', err);
  process.exit(1);
});
