;; ADD TO init.el

(setq openai-max-length 300)

(defun eglot-code-action-openai-gpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "text-davinci-002" openai-max-length))))

(defun eglot-code-action-openai-chatgpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "gpt-3.5-turbo" openai-max-length))))

(defun eglot-code-action-local-llm ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.localLlmStream (vector doc range))))

(defun eglot-code-action-stop-local-llm ()
  (interactive)
  (eglot-execute-command (eglot--current-server-or-lose) 'command.localLlmStreamStop '()))

(add-hook 'llm-mode-hook
          (lambda ()
            (define-key llm-mode-map (kbd "C-c l g") 'eglot-code-action-openai-gpt)
            (define-key llm-mode-map (kbd "C-c l c") 'eglot-code-action-openai-chatgpt)
            (define-key llm-mode-map (kbd "C-c l l") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c C-c") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c l s") 'eglot-code-action-stop-local-llm)
            (eglot-ensure)))

(add-to-list 'eglot-server-programs
             `(llm-mode . ("localhost" 5033)))
