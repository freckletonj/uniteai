import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import * as vscode from 'vscode';

import {
	LanguageClient,
	LanguageClientOptions,
	ServerOptions,
	TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

// Experiment with applying to all file types. May have unexpected results, so
// we may need a huge list of opted-in file types.
//   https://code.visualstudio.com/docs/languages/identifiers
const all_file_types = {
    documentSelector: [{scheme: 'file', language: '*'}],
    synchronize: {
        fileEvents: workspace.createFileSystemWatcher('**/*')
    }
};

export function activate(context: ExtensionContext) {
    // The server is implemented in python
    const serverExecutable = { command: 'uniteai_lsp', args: ['--stdio'] };

    const serverOptions: ServerOptions = {
        run: serverExecutable,
        debug: serverExecutable
    };

    // Options to control the language client
    const clientOptions: LanguageClientOptions = all_file_types;

    // Create the language client and start the client.
    client = new LanguageClient(
        'uniteai',
        'UniteAI',
        serverOptions,
        clientOptions
    );

    // Register the commands
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.stop', lspStop));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.exampleCounter', lspExampleCounter));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.localLlm', lspLocalLlm));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.transcribe', lspTranscribe));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.openaiGpt', lspOpenaiGpt));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.openaiChatgpt', lspOpenaiChatgpt));
    context.subscriptions.push(vscode.commands.registerCommand('uniteai.document', lspDocument));

    // Start the client and launch the server.
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
	if (!client) {
		return undefined;
	}
	return client.stop();
}

// Global stopping
async function lspStop() {
    const doc = vscode.window.activeTextEditor.document;
    client.sendRequest("workspace/executeCommand", {
        command: "command.stop",
        arguments: [{uri: doc.uri.toString()}]
    });
}

// Example Counter
async function lspExampleCounter() {
    const doc = vscode.window.activeTextEditor.document;
    const pos = vscode.window.activeTextEditor.selection.active;
    client.sendRequest("workspace/executeCommand", {
        command: "command.exampleCounter",
        arguments: [{uri: doc.uri.toString()}, {line: pos.line, character: pos.character}]
    });
}

// Document Chat
async function lspDocument() {
    const doc = vscode.window.activeTextEditor.document;
    const range = vscode.window.activeTextEditor.selection;
    client.sendRequest("workspace/executeCommand", {
        command: "command.document",
        arguments: [{uri: doc.uri.toString()}, {
            start: {line: range.start.line, character: range.start.character},
            end: {line: range.end.line, character: range.end.character}
        }]
    });
}

// Local LLM
async function lspLocalLlm() {
    const doc = vscode.window.activeTextEditor.document;
    const range = vscode.window.activeTextEditor.selection;
    client.sendRequest("workspace/executeCommand", {
        command: "command.localLlmStream",
        arguments: [{uri: doc.uri.toString()}, {
            start: {line: range.start.line, character: range.start.character},
            end: {line: range.end.line, character: range.end.character}
        }]
    });
}

// Transcription
async function lspTranscribe() {
    const doc = vscode.window.activeTextEditor.document;
    const pos = vscode.window.activeTextEditor.selection.active;
    client.sendRequest("workspace/executeCommand", {
        command: "command.transcribe",
        arguments: [{uri: doc.uri.toString()}, {line: pos.line, character: pos.character}]
    });
}

// OpenAI
async function lspOpenaiGpt() {
    const doc = vscode.window.activeTextEditor.document;
    const range = vscode.window.activeTextEditor.selection;
    client.sendRequest("workspace/executeCommand", {
        command: "command.openaiAutocompleteStream",
        arguments: [{uri: doc.uri.toString()}, {
            start: {line: range.start.line, character: range.start.character},
            end: {line: range.end.line, character: range.end.character}
        }, "FROM_CONFIG_COMPLETION", "FROM_CONFIG"]
    });
}

async function lspOpenaiChatgpt() {
    const doc = vscode.window.activeTextEditor.document;
    const range = vscode.window.activeTextEditor.selection;
    client.sendRequest("workspace/executeCommand", {
        command: "command.openaiAutocompleteStream",
        arguments: [{uri: doc.uri.toString()}, {
            start: {line: range.start.line, character: range.start.character},
            end: {line: range.end.line, character: range.end.character}
        }, "FROM_CONFIG_CHAT", "FROM_CONFIG"]
    });
}
