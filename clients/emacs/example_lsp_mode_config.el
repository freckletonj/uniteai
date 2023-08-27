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
  ((lambda ()
     (dolist (hook '((markdown-mode    . markdown)
                     (org-mode         . org)
                     (python-mode      . python)
                     (bibtex-mode      . bibtex)
                     (clojure-mode     . clojure)
                     (coffee-mode      . coffeescript)
                     (c-mode           . c)
                     (c++-mode         . cpp)
                     (csharp-mode      . csharp)
                     (css-mode         . css)
                     (diff-mode        . diff)
                     (dockerfile-mode  . dockerfile)
                     (fsharp-mode      . fsharp)
                     (go-mode          . go)
                     (groovy-mode      . groovy)
                     (html-mode        . html)
                     (web-mode         . html)  ; For HTML as well
                     (java-mode        . java)
                     (js-mode          . javascript)
                     (js2-mode         . javascriptreact)
                     (json-mode        . json)
                     (LaTeX-mode       . latex) ; Use LaTeX-mode, the AUCTeX mode for LaTeX
                     (less-css-mode    . less)
                     (lua-mode         . lua)
                     (makefile-mode    . makefile)
                     (objc-mode        . objective-c)
                     (perl-mode        . perl)
                     (php-mode         . php)
                     (text-mode        . plaintext)
                     (powershell-mode  . powershell)
                     (ess-mode         . r)  ; ESS mode for R
                     (ruby-mode        . ruby)
                     (rust-mode        . rust)
                     (scss-mode        . scss)
                     (sass-mode        . sass)
                     (sh-mode          . shellscript)
                     (sql-mode         . sql)
                     (swift-mode       . swift)
                     (typescript-mode  . typescript)
                     (TeX-mode         . tex)  ; For generic TeX files, it could be LaTeX or plain TeX
                     (nxml-mode        . xml)
                     (yaml-mode        . yaml)
                     ;; Additional modes
                     (sh-mode          . bash)
                     (toml-mode        . toml)))
       (let ((hook-symbol (car hook))
             (mode-string (cdr hook)))
         (when (fboundp hook-symbol)
           (add-hook hook-symbol (lambda ()
                                    (llm-mode 1)
                                    (lsp))))))))
  :init
  ;; Tell lsp-mode not to ask if you're ok with the server modifying the document.
  (setq lsp-restart 'auto-restart)
  :config
  (define-key lsp-command-map (kbd "M-'") 'lsp-execute-code-action)
  (setq lsp-enable-suggest-server-download nil)  ; if it can't launch the right LSP, don't pester
  )


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

            (define-key map (kbd "C-c l d") 'lsp-document)
            map))

;; set debug logs, but slows it down
(setq lsp-log-io t)

(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection "uniteai_lsp --stdio")
                  :major-modes '(python-mode markdown-mode org-mode)
                  :server-id 'llm-lsp
                  :priority -2
                  :add-on? t  ; run in parallel to other LSPs
                  ))

(lsp-register-client
 (make-lsp-client :new-connection (lsp-tcp-connection
                                   (lambda (port)
                                     `("uniteai_lsp" "--tcp" "--lsp_port" ,(number-to-string port))))
                  :priority -3
                  :major-modes '(python-mode markdown-mode org-mode)
                  :server-id 'llm-lsp-tcp
                  :add-on? t  ; run in parallel to other LSPs
                  ))
