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
  ((LaTeX-mode
    TeX-mode
    bibtex-mode
    c++-mode
    c-mode
    clojure-mode
    coffee-mode
    csharp-mode
    css-mode
    diff-mode
    dockerfile-mode
    ess-mode
    emacs-lisp-mode
    fsharp-mode
    go-mode
    groovy-mode
    html-mode
    java-mode
    js-mode
    js2-mode
    json-mode
    less-css-mode
    lua-mode
    makefile-mode
    markdown-mode
    nxml-mode
    objc-mode
    org-mode
    perl-mode
    php-mode
    powershell-mode
    python-mode
    ruby-mode
    rust-mode
    sass-mode
    scss-mode
    sh-mode
    sql-mode
    swift-mode
    text-mode
    toml-mode
    typescript-mode
    web-mode
    yaml-mode
    ) . (lambda ()
          (uniteai-mode 1)
          (lsp)))

  :init
  ;; Tell lsp-mode not to ask if you're ok with the server modifying the document.
  (setq lsp-restart 'auto-restart)

  :config
  ;; if it can't launch the right LSP, don't pester
  (setq lsp-enable-suggest-server-download nil)

  :bind
  ;; Code Actions
  ("M-'" . lsp-execute-code-action))


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

;; Document
(defun lsp-document ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* (
         (doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.document"
                   :arguments ,(vector doc range)))))

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


;; Local LLM
(defun lsp-text-to-speech-save ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* ((doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.textToSpeechSave"
                    :arguments ,(vector doc range)))))

;; Local LLM
(defun lsp-text-to-speech-play ()
  (interactive)
  (unless (region-active-p)
    (error "No region selected"))
  (let* ((doc (lsp--text-document-identifier))
         (range (list :start (lsp--point-to-position (region-beginning))
                      :end (lsp--point-to-position (region-end)))))
    (lsp-request "workspace/executeCommand"
                 `(:command "command.textToSpeechPlay"
                    :arguments ,(vector doc range)))))


(define-minor-mode uniteai-mode
  "Minor mode for interacting with UniteAI server."
  :lighter " UniteAI"
  :keymap (let ((map (make-sparse-keymap)))
            (define-key map (kbd "C-c l s") 'lsp-stop)
            (define-key map (kbd "C-c l e") 'lsp-example-counter)
            (define-key map (kbd "C-c l l") 'lsp-local-llm)
            (define-key map (kbd "C-c l v") 'lsp-transcribe)
            (define-key map (kbd "C-c l g") 'lsp-openai-gpt)
            (define-key map (kbd "C-c l c") 'lsp-openai-chatgpt)
            (define-key map (kbd "C-c l d") 'lsp-document)

            (define-key map (kbd "C-c l a s") 'lsp-text-to-speech-save)
            (define-key map (kbd "C-c l a p") 'lsp-text-to-speech-play)
            map))

(lsp-register-client
 (make-lsp-client

  ;; STDIO Mode
  :new-connection (lsp-stdio-connection `("uniteai_lsp" "--stdio"))

  ;; ;; TCP Mode
  ;; :new-connection (lsp-tcp-connection
  ;;                  (lambda (port)
  ;;                    `("uniteai_lsp" "--tcp" "--lsp_port" ,(number-to-string port))))

  :priority -3
  :major-modes '(python-mode markdown-mode org-mode)
  :server-id 'uniteai
  :add-on? t  ; run in parallel to other LSPs
  ))

(setq lsp-tcp-connection-timeout 5.0)  ; default was 2.0
