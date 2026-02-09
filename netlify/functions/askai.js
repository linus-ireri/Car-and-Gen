const axios = require("axios");

// --- Environment Variable Check ---
if (!process.env.OPENROUTER_API_KEY) {
  throw new Error("OPENROUTER_API_KEY is not configured. Please check your environment variables.");
}

// --- Constants ---
const RAG_SERVER_URL = process.env.RAG_SERVER_URL; // optional: if missing, we'll skip RAG and use LLM fallback
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GREETING_RESPONSES = {
  "who are you": "I am Civic, an AI assistant specializing in Kenyan legislation and policy information. How can I assist you today?",
  "hello": "Hello! I am Civic your legislative information assistant. How can I help you?",
  "hi": "Hi there! I am Civic, your legislative information assistant. How can I help you?",
  "hey": "Hello! I am Civic, your legislative information assistant. How can I help you?",
  "how are you": "I'm here to help you with questions about Kenyan legislation and policy. What would you like to know?",
  "good morning": "Good morning! I am Civic, your legislative information assistant. How can I help you?",
  "good afternoon": "Good afternoon! I am Civic, your legislative information assistant. How can I help you?",
  "good evening": "Good evening! I am Civic, your legislative information assistant. How can I help you?"
};

// --- Helper Functions ---

/**
 * Normalizes a string by converting it to lowercase, removing special characters, and trimming whitespace.
 * @param {string} text - The text to normalize.
 * @returns {string} The normalized text.
 */
function normalize(text) {
  return text.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Handles rule-based responses for common greetings.
 * @param {string} normalizedMessage - The normalized user message.
 * @returns {object|null} A response object or null if no greeting is matched.
 */
function getGreetingResponse(normalizedMessage) {
  if (GREETING_RESPONSES[normalizedMessage]) {
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reply: GREETING_RESPONSES[normalizedMessage],
        context: [],
        source: "rule-based"
      }),
    };
  }
  return null;
}

/**
 * Queries the RAG server for an answer.
 * @param {string} userMessage - The user's message.
 * @returns {Promise<object|null>} The RAG server's response or null on error.
 */
async function queryRagServer(userMessage) {
  try {
    if (!RAG_SERVER_URL) {
      return null;
    }
    // Use the configured endpoint as-is; expected to be '/ask'
    const ragUrl = RAG_SERVER_URL;
    console.log("Querying RAG server at:", ragUrl);
    const ragResponse = await axios.post(
      ragUrl,
      { question: userMessage },
      { timeout: 4000 } // keep RAG call short to fit Netlify's 10s limit
    );
    return ragResponse.data;
  } catch (error) {
    console.error("RAG server error:", error.message);
    return null;
  }
}

/**
 * Queries the LLM with context from the RAG server.
 * @param {string} userMessage - The user's message.
 * @param {Array<string>} context - The context from the RAG server.
 * @returns {Promise<object>} The LLM's response.
 */
async function queryLlmWithContext(userMessage, context) {
  const systemPrompt = `You are Civic, an AI assistant specializing in Kenyan legislation and policy information. Your responses must:
1. Be based ONLY on the retrieved context provided
2. Cite specific acts, bills, or policies when they are referenced
3. Say "I don't have enough information about that in my knowledge base" if the context doesn't contain relevant information
4. Be clear and precise, avoiding speculation or inference
5. Focus solely on legislative and policy information
Never identify yourself as an AI model or mention any model providers. Maintain a professional, informative tone.`;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Retrieved context: ${context.join(" ")}` },
    { role: "user", content: userMessage }
  ];

  try {
    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      { model: "mistralai/mistral-small-3.2-24b-instruct:free", messages },
      {
        headers: {
          "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 6000
      }
    );
    const answer = response.data.choices?.[0]?.message?.content?.trim();
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reply: answer || "Sorry, I do not have official information on that topic.",
        context: context,
        source: "rag+llm"
      }),
    };
  } catch (error) {
    console.error("LLM with context error:", error.message);
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reply: "Sorry, I do not have official information on that topic.",
        context: context,
        source: "rag+llm-fallback"
      }),
    };
  }
}

/**
 * Queries the LLM as a fallback when the RAG server fails.
 * @param {string} userMessage - The user's message.
 * @returns {Promise<object>} The LLM's response.
 */
async function queryLlmFallback(userMessage) {
  const systemPrompt = `You are Civic, an AI assistant specializing in Kenyan legislation and policy information. Since no context is available for this query:
1. Politely explain that you can only provide information about legislation and policy that is in your knowledge base
2. Suggest that the user try rephrasing their question to focus on specific acts, bills, or policies
3. Maintain a professional, helpful tone
Never identify yourself as an AI model or mention any model providers.`;

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userMessage }
  ];

  try {
    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "mistralai/mistral-7b-instruct:free",
        messages
      },
      {
        headers: {
          "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 3000
      }
    );
    const answer = response.data.choices?.[0]?.message?.content?.trim();
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reply: answer || "Sorry, I do not have official information on that topic.",
        context: [],
        source: "llm-fallback"
      }),
    };
  } catch (error) {
    console.error("LLM fallback error:", error.message);
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reply: "I'm experiencing high traffic right now and can't answer this question at the moment. Please try again in a few minutes!",
        context: [],
        source: "rule-fallback"
      }),
    };
  }
}

// --- Main Handler ---

exports.handler = async function (event, context) {
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: "Method Not Allowed. Use POST instead." }),
    };
  }

  try {
    const body = JSON.parse(event.body || "{}");
    const userMessage = body.message?.trim();

    if (!userMessage || typeof userMessage !== "string") {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Invalid request format. Expected a 'message' field with text." }),
      };
    }

    const normalizedMessage = normalize(userMessage);

    // 1) Greetings: if RAG is configured, use lightweight rule-based reply;
    // if RAG is not configured/likely down, respond via LLM so it introduces capabilities.
    const greetingResponse = getGreetingResponse(normalizedMessage);
    if (greetingResponse) {
      if (!RAG_SERVER_URL) {
        return await queryLlmFallback(userMessage);
      }
      return greetingResponse;
    }

    // 2) Query RAG server at /ask
    const ragData = await queryRagServer(userMessage);
    if (ragData) {
      if (ragData.answer) {
        return {
          statusCode: 200,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reply: ragData.answer, context: ragData.context || [], source: "rag" }),
        };
      }
      if (ragData.context && Array.isArray(ragData.context) && ragData.context.length > 0) {
        // 3) LLM with context if no direct answer
        return await queryLlmWithContext(userMessage, ragData.context);
      }
    }

    // 4) Fallback LLM without context
    return await queryLlmFallback(userMessage);

  } catch (error) {
    console.error("Unexpected error:", error.message);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Unexpected error", details: error.message }),
    };
  }
};