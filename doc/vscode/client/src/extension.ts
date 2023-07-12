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

export function activate(context: ExtensionContext) {
  // The server is implemented in python
  const serverExecutable = { command: 'uniteai_lsp', args: ['--stdio'] };

  const serverOptions: ServerOptions = {
    run: serverExecutable,
    debug: serverExecutable
  };

  // Options to control the language client
  const clientOptions: LanguageClientOptions = {
    // Register the server for plain text documents
    documentSelector: [{ scheme: 'file', language: 'plaintext' }],
    synchronize: {}
  };

  // Create the language client and start the client.
  client = new LanguageClient(
    'uniteai',
    'UniteAI',
    serverOptions,
    clientOptions
  );

  // // Register the commands
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.stop', lspStop));
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.exampleCounter', lspExampleCounter));
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.localLlm', lspLocalLlm));
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.transcribe', lspTranscribe));
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.openaiGpt', lspOpenaiGpt));
  // context.subscriptions.push(vscode.commands.registerCommand('uniteai.openaiChatgpt', lspOpenaiChatgpt));

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
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  client.sendRequest("workspace/executeCommand", {
    command: "command.stop",
    arguments: [doc]
  });
}

// Example Counter
async function lspExampleCounter() {
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  const pos = vscode.window.activeTextEditor.selection.active;
  client.sendRequest("workspace/executeCommand", {
    command: "command.exampleCounter",
    arguments: [doc, pos]
  });
}

// Local LLM
async function lspLocalLlm() {
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  const range = new vscode.Range(vscode.window.activeTextEditor.selection.start, vscode.window.activeTextEditor.selection.end);
  client.sendRequest("workspace/executeCommand", {
    command: "command.localLlmStream",
    arguments: [doc, range]
  });
}

// Transcription
async function lspTranscribe() {
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  const pos = vscode.window.activeTextEditor.selection.active;
  client.sendRequest("workspace/executeCommand", {
    command: "command.transcribe",
    arguments: [doc, pos]
  });
}

// OpenAI
async function lspOpenaiGpt() {
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  const range = new vscode.Range(vscode.window.activeTextEditor.selection.start, vscode.window.activeTextEditor.selection.end);
  client.sendRequest("workspace/executeCommand", {
    command: "command.openaiAutocompleteStream",
    arguments: [doc, range, "FROM_CONFIG_COMPLETION", "FROM_CONFIG"]
  });
}

async function lspOpenaiChatgpt() {
  const doc = await vscode.workspace.openTextDocument(vscode.window.activeTextEditor.document.uri);
  const range = new vscode.Range(vscode.window.activeTextEditor.selection.start, vscode.window.activeTextEditor.selection.end);
  client.sendRequest("workspace/executeCommand", {
    command: "command.openaiAutocompleteStream",
    arguments: [doc, range, "FROM_CONFIG_CHAT", "FROM_CONFIG"]
  });
}
