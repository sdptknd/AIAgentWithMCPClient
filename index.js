import { config } from 'dotenv';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import readline from 'readline/promises';
import { GoogleGenAI } from '@google/genai';
import mcpConfig from './mcpConfig.json' with { type: 'json' };

config();

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const MCPClient = class {
    constructor(name, transport) {
        this.mcp = new Client({
            name: name || "Client-" + Math.random().toString(36).substring(2, 7),
            version: "1.0.0",
        });
        this.tools = null;
        this.initialised = false;
        this.transport = transport
    }

    async initiate() {
        if (this.initialised) return;
        await this.mcp.connect(this.transport);
        this.initialised = true;
    }

    async getTool() {
        if (!this.initialised) {
            await this.initiate();
        }
        if (!this.tools) {
            this.tools = await (await this.mcp.listTools()).tools;
            // console.log("Tools available:", JSON.stringify(this.tools, null, 2));
        }
        return this.tools;
    }

    async callTool(toolName, args) {
        // console.log(`Calling tool: ${toolName} with args:`, JSON.stringify(args));
        const response = await rl.question(`\nType yes to confirm calling tool ${toolName} with args ${JSON.stringify(args)} or anything else to cancel: `);

        if (response !== 'yes') {
            console.log("Tool call cancelled.");
            return { success: false, response: null };
        }

        return { success: true, response: await this.mcp.callTool({ name: toolName, arguments: args }) };
    }

    async cleanup() {
        await this.mcp.close();
    }
}


const MCPAgent = class {
    constructor(googleLLM, mcpClients) {
        this.googleLLM = googleLLM;
        this.mcpClients = mcpClients || [];
        this._cachedTools = null;
    }

    async loadTools() {
        if (this._cachedTools == null) {
            this._cachedTools = {};
            for (let mcpClient of this.mcpClients) {
                const tools = await mcpClient.getTool();
                tools.map(tool => {
                    const parameters = {};
                    Object.keys(tool.inputSchema).filter(key => !['additionalProperties', '$schema'].includes(key)).forEach(key => {
                        parameters[key] = tool.inputSchema[key];
                    });
                    this._cachedTools[tool.name] = {
                        definition: {
                            name: tool.name,
                            description: tool.description,
                            parameters
                        }, client: mcpClient
                    };
                });
            }
        }
        return Object.values(this._cachedTools).map(tool => ({ functionDeclarations: [tool.definition] }));
    }

    async processQuery(query, role = "user", savedContent = []) {
        const currentContents = [...savedContent, { parts: [{ text: query }], role }];
        let response;
        while (true) {
            response = await this.googleLLM.models.generateContent({
                // model: 'gemini-2.0-flash-lite',
                model: 'gemini-2.0-flash',
                contents: currentContents,
                config: {
                    tools: await this.loadTools()
                }
            });
            currentContents.push(response.candidates[0].content);

            if (response.candidates[0].content.parts.every((part) => !('functionCall' in part))) {
                break;
            }

            const functionResponseParts = [];

            for (let part of response.candidates[0].content.parts) {
                if (!part.functionCall) continue;

                const functionClient = this._cachedTools[part.functionCall.name].client;
                const { response: toolResponse, success } = await functionClient.callTool(part.functionCall.name, part.functionCall.args);
                // console.log(`Calling tool: ${part.functionCall.name} with args:`, JSON.stringify(part.functionCall.args));
                if (success) {
                    functionResponseParts.push({
                        functionResponse: {
                            name: part.functionCall.name,
                            response: toolResponse
                        }
                    });
                } else {
                    functionResponseParts.push({
                        functionResponse: {
                            name: part.functionCall.name,
                            response: { content: { type: "text", "text": "Tool/Function call failed or was cancelled." } }
                        }
                    });
                }
            }
            currentContents.push({
                parts: functionResponseParts,
                role: "user"
            });
        }

        return [response, currentContents];
    }

    async chatLoop() {
        let history = [{ parts: [{ text: "You are a helpful AI assistant. USE AVAILABLE TOOLS FOR ANY INFORMATION YOU NEED BEFORE ASKING FOR THAT" }], role: "user" }];
        let response;

        try {
            console.log("\nMCP Agent Started!");
            console.log("Type your queries or 'quit' to exit.");

            while (true) {
                const message = await rl.question("\nYou: ");
                if (message.toLowerCase() === "quit") {
                    // console.log("history: " + JSON.stringify(await this.llmChat.getHistory(), null, 2));
                    break;
                }
                [response, history] = await this.processQuery(message, "user", history);
                console.log("\nAI: " + response.candidates[0].content.parts[0].text);
            }
        } finally {
            rl.close();
        }
    }
}

const getMCPClients = (mcpConfig) => {
    return Object.entries(mcpConfig).map(([name, config]) => {
        if (!("url" in config)) {
            return new MCPClient(name, new StdioClientTransport({
                command: config.command,
                args: config.args,
            }));
        } else if (config.type === 'http') {
            return new MCPClient(name, new StreamableHTTPClientTransport(config.url));
        } else if (config.type === 'sse') {
            return new MCPClient(name, new SSEClientTransport(config.url));
        } else {
            throw new Error(`Unknown transport type: ${config.type}`);
        }
    })
}

const main = async () => {
    try {
        const llm = new GoogleGenAI({
            apiKey: process.env.GOOGLE_API_KEY,
            maxTokens: 1000,
            temperature: 0.7,
        });

        const agent = new MCPAgent(llm, getMCPClients(mcpConfig));
        await agent.chatLoop();
        process.exit(0);
    } catch (error) {
        console.error("Error in main:", error);
    }
}
main().catch(console.error);
