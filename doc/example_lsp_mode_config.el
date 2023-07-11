;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LSP-MODE CONFIGURATION
;;
;;     add to init.el
;;
;; `lsp-mode` excels over eglot in that you can run it in parallel alongside
;; other LSPs.


(use-package lsp-mode
  :ensure t
  :commands lsp
  :hook
  ((markdown-mode org-mode python-mode) . (lambda ()
                                            (llm-mode 1)
                                            (lsp)))
  :init
  ;; Tell lsp-mode not to ask if you're ok with the server modifying the document.
  (setq lsp-restart 'auto-restart)
  :config
  (define-key lsp-command-map (kbd "M-'") 'lsp-execute-code-action))

;;;;;;;;;;

;; Global stopping
(defun lsp-stop ()
  (interactive)
  (let* ((doc (lsp--text-document-identifier)))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.stop"
                    :arguments ,(vector doc)))))

;; Example Counter
(defun lsp-example-counter ()
  (interactive)
  (let* ((doc (lsp--text-document-identifier))
         (pos (lsp--cur-position)))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.exampleCounter"
                    :arguments ,(vector doc pos)))))

;; Local LLM
(defun lsp-local-llm ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* ((doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.localLlmStream"
                    :arguments ,(vector doc range)))))

;; Transcription
(defun lsp-transcribe ()
  (interactive)
  (let* ((doc (lsp--text-document-identifier))
         (pos (lsp--cur-position)))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.transcribe"
                   :arguments ,(vector doc pos)))))

;; OpenAI
(defun lsp-openai-gpt ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* (
         (doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.openaiAutocompleteStream"
                   :arguments ,(vector doc range "FROM_CONFIG_COMPLETION" "FROM_CONFIG")))))

(defun lsp-openai-chatgpt ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* (
         (doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.openaiAutocompleteStream"
                   :arguments ,(vector doc range "FROM_CONFIG_CHAT" "FROM_CONFIG")))))



(define-minor-mode llm-mode
  "Minor mode for interacting with LLM server."
  :lighter " LLM"
  :keymap (let ((map (make-sparse-keymap)))
            (define-key map (kbd "C-c l s") 'lsp-stop)

            (define-key map (kbd "C-c l e") 'lsp-example-counter)

            (define-key map (kbd "C-c l l") 'lsp-local-llm)

            (define-key map (kbd "C-c l v") 'lsp-transcribe)

            (define-key map (kbd "C-c l g") 'lsp-openai-gpt)
            (define-key map (kbd "C-c l c") 'lsp-openai-chatgpt)
            map))

;; set debug logs, but slows it down
(setq lsp-log-io t)

(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection "python lsp_server.py --stdio")
                  :major-modes '(python-mode markdown-mode org-mode)
                  :server-id 'llm-lsp
                  :priority -2
                  :add-on? t  ; run in parallel to other LSPs
                  ))

(lsp-register-client
 (make-lsp-client :new-connection (lsp-tcp-connection
                                   (lambda (port)
                                     `("python" "lsp_server.py" "--tcp" "--lsp_port" ,(number-to-string port))))
                  :priority -3
                  :major-modes '(python-mode markdown-mode org-mode)
                  :server-id 'llm-lsp-tcp
                  :add-on? t  ; run in parallel to other LSPs
                  ))
