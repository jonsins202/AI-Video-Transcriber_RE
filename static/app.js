/* ────────────────────────────────────────────────────────────
   AI Video Transcriber · app.js
   ──────────────────────────────────────────────────────────── */

class VideoTranscriber {
  constructor() {
    this.currentTaskId  = null;
    this.eventSource    = null;
    this.apiBase        = '/api';
    this.currentLang    = 'en';

    /* Smart progress simulation */
    this.sp = {
      enabled: false, current: 0, target: 15,
      lastServer: 0, interval: null, startTime: null, stage: 'preparing'
    };

    this.i18n = {
      en: {
        title:                   'AI Video Transcriber',
        subtitle:                'Turn any video or audio into clean transcripts, summaries and translations — automatically.',
        hero_badge:              '30+ platforms supported',
        video_url_placeholder:   'Paste YouTube, TikTok, Bilibili or other platform video URLs...',
        start_transcription:     'Transcribe',
        ai_settings:             'AI Settings',
        model_base_url:          'Model API Base URL',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            'Fetch',
        model_select:            'Model',
        model_default:           '— use server default —',
        summary_language:        'Summary Language',
        oral_cleanup:            'Oral cleanup',
        oral_cleanup_hint:       'Remove filler words & smooth sentences (not verbatim)',
        cleaning_oral:           'Cleaning up speech…',
        spk_num:                 'Speakers',
        spk_num_auto:            'Auto',
        spk_num_hint:            'Speaker count (optional): leave empty for auto-detect; fill in when known (e.g. 2 for interviews) for better accuracy',
        speaker_rename_title:    'Name the speakers',
        speaker_rename_hint:     'Fill real names for labels (empty keeps the label); transcript updates instantly, summary keeps old labels',
        speaker_apply:           'Apply',
        speaker_apply_failed:    'Failed to apply speaker names',
        processing_progress:     'Processing',
        preparing:               'Preparing…',
        transcript_text:         'Transcript',
        intelligent_summary:     'AI Summary',
        translation:             'Translation',
        download_transcript:     'Transcript',
        download_translation:    'Translation',
        download_summary:        'Summary',
        empty_hint:              'Paste a video URL or drop a file above and let AI do the heavy lifting.',
        footer_text:             'Built on the open-source <a href="https://github.com/wendy7756/AI-Video-Transcriber" target="_blank">AI-Video-Transcriber</a> — thanks to the original author wendy7756',
        total_time:              'Total',
        finished_at:             'Done at',
        dur_h:                   'h',
        dur_m:                   'm',
        dur_s:                   's',
        stage_subtitle:          'Subtitles',
        stage_download:          'Download',
        stage_transcribe:        'Transcribe',
        stage_optimize:          'Optimize',
        stage_cleanup:           'Cleanup',
        stage_summarize:         'Summarize',
        stage_translate:         'Translate',
        upload_or:               'Drag & drop a file here — or browse your computer',
        processing:              'Processing…',
        downloading_video:       'Downloading audio…',
        parsing_video:           'Parsing video info…',
        transcribing_audio:      'Transcribing audio…',
        optimizing_transcript:   'Optimizing transcript…',
        generating_summary:      'Generating summary…',
        detecting_subtitles:     'Detecting subtitles…',
        subtitle_found:          'Subtitles found! Processing text…',
        no_subtitle:             'No subtitles found, downloading audio…',
        cache_hit_raw:           'Local transcript cache hit — skipping download & transcription',
        cache_hit_audio:         'Local audio cache hit — skipping download',
        mode_subtitle:           '⚡ Subtitle',
        mode_whisper:            '🎙 Whisper',
        completed:               'Done!',
        error_invalid_url:       'Please enter a valid video URL',
        error_processing_failed: 'Processing failed: ',
        error_no_download:       'No file available for download',
        error_download_failed:   'Download failed: ',
        fetching_models:         'Fetching models…',
        models_loaded:           (n) => `${n} models loaded`,
        models_error:            'Failed to fetch models',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        'Upload files',
        error_upload_type:       'Unsupported file type',
        error_upload_empty:      'File is empty',
        error_upload_size:       (mb) => `File exceeds ${mb} MB limit`,
        batch_mode:              'Batch',
        batch_editor_title:      'Batch transcription',
        batch_add_row:           'Add row',
        batch_clear:             'Clear',
        batch_exit:              'Exit',
        batch_submit:            (n) => `Start batch (${n})`,
        batch_max_hint:          'Up to 10 URLs',
        batch_truncated_notice:  'Up to 10 URLs supported — truncated',
        batch_invalid_url:       'Enter a valid URL',
        batch_duplicate:         'Duplicate URL',
        batch_url_placeholder:   'https://…',
        batch_remove:            'Remove row',
        batch_skipped:           'Skipped: ',
        task_queue_title:        'Task queue',
        stat_total:              'Total',
        stat_done:               'Done',
        stat_failed:             'Failed',
        stat_running:            'Running',
        stat_queued:             'Queued',
        stat_interrupted:        'Interrupted',
        task_view:               'View',
        task_dl_script:          'Transcript',
        task_dl_summary:         'Summary',
        task_retry:              'Retry',
        task_requeue:            'Requeue',
        task_cancel:             'Cancel',
        task_merge:              'Merge & download',
        task_merge_disabled_hint:'Merge available when all tasks finish',
        task_merge_done:         'Merged document ready — downloading',
        task_merge_failed:       'Merge failed: ',
        task_action_failed:      'Action failed: ',
        task_load_failed:        'Failed to load task list',
        task_batch_group:        (n) => `Batch · ${n} tasks`,
        task_meta_cleanup:       'Oral cleanup',
        task_meta_spk:           (n) => `Speakers ${n}`,
        chip_queued:             'Queued',
        chip_running:            'Running',
        chip_completed:          'Done',
        chip_error:              'Failed',
        chip_interrupted:        'Interrupted',
        chip_cancelled:          'Cancelled',
        api_history_title:       'API config history',
        api_history_empty:       'No saved configs yet',
        api_history_delete:      'Remove this config',
        api_history_applied:     'Config applied — fetching models…',
      },
      zh: {
        title:                   'AI 视频转录器',
        subtitle:                '粘贴视频链接或上传本地文件，自动完成转录、翻译与 AI 摘要。',
        hero_badge:              '支持 30+ 平台',
        video_url_placeholder:   '请输入视频链接…',
        start_transcription:     '开始转录',
        ai_settings:             'AI 设置',
        model_base_url:          'Model API 地址',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            '获取',
        model_select:            '模型',
        model_default:           '— 使用服务器默认 —',
        summary_language:        '摘要语言',
        oral_cleanup:            '口语清理',
        oral_cleanup_hint:       '删语气词并顺句（非逐字稿）',
        cleaning_oral:           '正在清理口语…',
        spk_num:                 '说话人数',
        spk_num_auto:            '自动',
        spk_num_hint:            '说话人分离人数（可选）：留空自动检测；已知人数时填写（如双人访谈填 2）识别更准',
        speaker_rename_title:    '说话人命名',
        speaker_rename_hint:     '为标签填真名（留空保留标签）；文稿即时更新，摘要仍用原标签',
        speaker_apply:           '应用',
        speaker_apply_failed:    '说话人命名应用失败',
        processing_progress:     '处理进度',
        preparing:               '准备中…',
        transcript_text:         '转录文本',
        intelligent_summary:     '智能摘要',
        translation:             '翻译',
        download_transcript:     '转录',
        download_translation:    '翻译',
        download_summary:        '摘要',
        empty_hint:              '在上方粘贴视频链接或拖放文件，让 AI 来处理一切。',
        footer_text:             '基于开源项目 <a href="https://github.com/wendy7756/AI-Video-Transcriber" target="_blank">AI-Video-Transcriber</a> 二次开发 · 感谢原作者 wendy7756',
        total_time:              '总耗时',
        finished_at:             '完成于',
        dur_h:                   '小时',
        dur_m:                   '分',
        dur_s:                   '秒',
        stage_subtitle:          '字幕',
        stage_download:          '下载',
        stage_transcribe:        '转录',
        stage_optimize:          '优化',
        stage_cleanup:           '口语清理',
        stage_summarize:         '摘要',
        stage_translate:         '翻译',
        upload_or:               '拖放文件到此处 — 或从本地选择',
        processing:              '处理中…',
        downloading_video:       '正在下载音频…',
        parsing_video:           '正在解析视频信息…',
        transcribing_audio:      '正在转录音频…',
        optimizing_transcript:   '正在优化转录文本…',
        generating_summary:      '正在生成摘要…',
        detecting_subtitles:     '正在检测字幕…',
        subtitle_found:          '字幕获取成功！正在处理文本…',
        no_subtitle:             '未找到字幕，正在下载音频…',
        cache_hit_raw:           '命中本地转录缓存，跳过下载与转录',
        cache_hit_audio:         '命中本地音频缓存，跳过下载',
        mode_subtitle:           '⚡ 字幕模式',
        mode_whisper:            '🎙 Whisper 模式',
        completed:               '处理完成！',
        error_invalid_url:       '请输入有效的视频链接',
        error_processing_failed: '处理失败：',
        error_no_download:       '没有可下载的文件',
        error_download_failed:   '下载失败：',
        fetching_models:         '正在获取模型列表…',
        models_loaded:           (n) => `已加载 ${n} 个模型`,
        models_error:            '获取模型失败',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        '上传文件',
        error_upload_type:       '不支持的文件类型',
        error_upload_empty:      '文件为空',
        error_upload_size:       (mb) => `文件超过 ${mb} MB 限制`,
        batch_mode:              '批量模式',
        batch_editor_title:      '批量转录',
        batch_add_row:           '＋添加一行',
        batch_clear:             '清空',
        batch_exit:              '取消',
        batch_submit:            (n) => `开始批量转录 (${n})`,
        batch_max_hint:          '最多支持 10 个链接',
        batch_truncated_notice:  '最多支持 10 个链接，已截断',
        batch_invalid_url:       '请输入有效的视频链接',
        batch_duplicate:         '批内重复链接',
        batch_url_placeholder:   'https://…',
        batch_remove:            '移除该行',
        batch_skipped:           '已跳过：',
        task_queue_title:        '任务队列',
        stat_total:              '共',
        stat_done:               '完成',
        stat_failed:             '失败',
        stat_running:            '进行中',
        stat_queued:             '排队',
        stat_interrupted:        '已中断',
        task_view:               '查看',
        task_dl_script:          '下载转录',
        task_dl_summary:         '下载摘要',
        task_retry:              '重试',
        task_requeue:            '重新入队',
        task_cancel:             '取消',
        task_merge:              '生成合并文档',
        task_merge_disabled_hint:'全部任务结束后可合并',
        task_merge_done:         '合并文档已生成，开始下载',
        task_merge_failed:       '合并失败：',
        task_action_failed:      '操作失败：',
        task_load_failed:        '任务列表加载失败',
        task_batch_group:        (n) => `批量任务 · ${n} 个`,
        task_meta_cleanup:       '口语清理',
        task_meta_spk:           (n) => `说话人数 ${n}`,
        chip_queued:             '排队中',
        chip_running:            '处理中',
        chip_completed:          '已完成',
        chip_error:              '失败',
        chip_interrupted:        '已中断',
        chip_cancelled:          '已取消',
        api_history_title:       'API 配置历史',
        api_history_empty:       '还没有保存过的配置',
        api_history_delete:      '删除此配置',
        api_history_applied:     '已填入历史配置，正在获取模型…',
      }
    };

    this._initElements();
    this._bindEvents();
    this._loadSettings();
    this._recordApiHistory(); // 页面加载即把当前已保存的配置补录进历史（幂等）
    this._switchLang(this._savedUiLang || 'zh');
    this._startTaskListPolling();
  }

  /* ── Elements ─────────────────────────────────────────── */
  _initElements() {
    this.form               = document.getElementById('videoForm');
    this.videoUrlInput      = document.getElementById('videoUrl');
    this.submitBtn          = document.getElementById('submitBtn');
    this.summaryLangSel     = document.getElementById('summaryLanguage');
    this.oralCleanupInput   = document.getElementById('oralCleanup');
    this.spkNumInput        = document.getElementById('spkNum');
    this.langToggle         = document.getElementById('langToggle');
    this.langText           = document.getElementById('langText');
    this.errorBanner        = document.getElementById('errorBanner');
    this.errorMsg           = document.getElementById('errorMsg');
    this.emptyState         = document.getElementById('emptyState');
    this.progressPanel      = document.getElementById('progressPanel');
    this.modeBadge          = document.getElementById('modeBadge');
    this.progressStatus     = document.getElementById('progressStatus');
    this.progressFill       = document.getElementById('progressFill');
    this.progressMessage    = document.getElementById('progressMessage');
    this.resultsPanel       = document.getElementById('resultsPanel');
    this.scriptContent      = document.getElementById('scriptContent');
    this.summaryContent     = document.getElementById('summaryContent');
    this.translationContent = document.getElementById('translationContent');
    this.dlScript           = document.getElementById('downloadScript');
    this.dlTranslation      = document.getElementById('downloadTranslation');
    this.dlSummary          = document.getElementById('downloadSummary');
    this.translationTabBtn  = document.getElementById('translationTabBtn');
    this.tabBtns            = document.querySelectorAll('.tab-btn');
    this.tabPanes           = document.querySelectorAll('.tab-pane');
    // settings
    this.settingsToggle     = document.getElementById('settingsToggle');
    this.settingsBody       = document.getElementById('settingsBody');
    this.modelBaseUrl       = document.getElementById('modelBaseUrl');
    this.apiKeyInput        = document.getElementById('apiKeyInput');
    this.fetchModelsBtn     = document.getElementById('fetchModelsBtn');
    this.fetchStatus        = document.getElementById('fetchStatus');
    this.modelSelect        = document.getElementById('modelSelect');
    this.fetchIcon          = document.getElementById('fetchIcon');
    this.uploadZone         = document.getElementById('uploadZone');
    this.uploadPickBtn      = document.getElementById('uploadPickBtn');
    this.fileInput          = document.getElementById('fileInput');
    this.uploadMaxMb        = 200;
    this._allowedUploadExts = new Set(['.txt', '.mp3', '.mp4', '.m4a', '.wav', '.webm', '.mkv', '.ogg', '.flac']);
    // batch mode + task list
    this.singleUrlWrap   = document.getElementById('singleUrlWrap');
    this.batchToggle     = document.getElementById('batchToggle');
    this.batchEditor     = document.getElementById('batchEditor');
    this.batchRowsEl     = document.getElementById('batchRows');
    this.batchNotice     = document.getElementById('batchNotice');
    this.batchAddBtn     = document.getElementById('batchAddBtn');
    this.batchClearBtn   = document.getElementById('batchClearBtn');
    this.batchExitBtn    = document.getElementById('batchExitBtn');
    this.taskListPanel   = document.getElementById('taskListPanel');
    this.taskListStats   = document.getElementById('taskListStats');
    this.taskListBody    = document.getElementById('taskListBody');
    this.toastEl         = document.getElementById('toast');
    this.apiHistoryMenu  = document.getElementById('apiHistoryMenu');
    this.settingsRow     = document.querySelector('.settings-row');
    this.batchOpen       = false;
    this._loading        = false;
    this._taskPollTimer  = null;
    this._toastTimer     = null;
    this._lastTasks      = [];
  }

  /* ── Events ───────────────────────────────────────────── */
  _bindEvents() {
    this.form.addEventListener('submit', (e) => {
      e.preventDefault();
      if (this.batchOpen) this._submitBatch();
      else this._startTranscription();
    });

    // Batch mode: toggle + paste detection + editor controls
    this.batchToggle.addEventListener('click', () => {
      if (this.batchOpen) this._closeBatchEditor();
      else this._openBatchEditor([]);
    });
    this.videoUrlInput.addEventListener('paste', (e) => this._onUrlPaste(e));
    this.batchAddBtn.addEventListener('click', () => this._addBatchRow(''));
    this.batchClearBtn.addEventListener('click', () => { this._clearBatchRows(); this._addBatchRow(''); });
    this.batchExitBtn.addEventListener('click', () => this._closeBatchEditor());
    this.batchRowsEl.addEventListener('input', (e) => {
      if (e.target.classList.contains('batch-url-input')) this._validateBatch();
    });
    this.batchRowsEl.addEventListener('click', (e) => {
      const rm = e.target.closest('.batch-remove');
      if (rm) {
        rm.closest('.batch-row').remove();
        this._renumberBatchRows();
        this._validateBatch();
      }
    });
    // Task list row actions (event delegation)
    this.taskListBody.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-task-action]');
      if (!btn) return;
      const id = btn.dataset.taskId;
      const act = btn.dataset.taskAction;
      if (act === 'view') this._viewTask(id);
      else if (act === 'dl-script') this._downloadTaskFile(id, 'script');
      else if (act === 'dl-summary') this._downloadTaskFile(id, 'summary');
      else if (act === 'requeue') this._taskAction(id, 'requeue');
      else if (act === 'cancel') this._taskAction(id, 'cancel');
    });
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') this._loadTaskList();
    });

    this.langToggle.addEventListener('click', () => {
      this._switchLang(this.currentLang === 'en' ? 'zh' : 'en');
      this._saveSettings();
    });

    // Settings toggle
    this.settingsToggle.addEventListener('click', () => {
      const open = this.settingsBody.classList.toggle('open');
      this.settingsToggle.classList.toggle('open', open);
    });

    // Fetch models
    this.fetchModelsBtn.addEventListener('click', () => this._fetchModels());

    // Auto-fetch when both fields filled (debounced)
    const debouncedFetch = this._debounce(() => {
      if (this.modelBaseUrl.value.trim() && this.apiKeyInput.value.trim()) this._fetchModels();
    }, 900);
    this.modelBaseUrl.addEventListener('input', debouncedFetch);
    this.apiKeyInput.addEventListener('input', debouncedFetch);

    // Persist settings
    [this.modelBaseUrl, this.apiKeyInput, this.modelSelect, this.summaryLangSel, this.oralCleanupInput, this.spkNumInput].forEach(el => {
      el.addEventListener('change', () => this._saveSettings());
    });

    // API 历史记录：聚焦地址输入框弹出，点击外部关闭
    this.modelBaseUrl.addEventListener('focus', () => this._showApiHistory());
    document.addEventListener('mousedown', (e) => {
      if (this.apiHistoryMenu.hidden) return;
      if (e.target === this.modelBaseUrl || this.apiHistoryMenu.contains(e.target)) return;
      this._hideApiHistory();
    });

    // Tabs
    this.tabBtns.forEach(btn => {
      btn.addEventListener('click', () => this._switchTab(btn.dataset.tab));
    });

    // Downloads
    this.dlScript.addEventListener('click',      () => this._downloadFile('script'));
    this.dlTranslation.addEventListener('click', () => this._downloadFile('translation'));
    this.dlSummary.addEventListener('click',     () => this._downloadFile('summary'));

    if (this.uploadPickBtn && this.fileInput && this.uploadZone) {
      this.uploadPickBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.fileInput.click();
      });
      this.uploadZone.addEventListener('click', (e) => {
        if (e.target === this.uploadPickBtn || this.uploadPickBtn.contains(e.target)) return;
        this.fileInput.click();
      });
      this.fileInput.addEventListener('change', () => {
        const f = this.fileInput.files && this.fileInput.files[0];
        this.fileInput.value = '';
        if (f) this._startFileUpload(f);
      });
      ['dragenter', 'dragover'].forEach((ev) => {
        this.uploadZone.addEventListener(ev, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.uploadZone.classList.add('dragover');
        });
      });
      this.uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        if (!this.uploadZone.contains(e.relatedTarget)) {
          this.uploadZone.classList.remove('dragover');
        }
      });
      this.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) this._startFileUpload(f);
      });
    }
  }

  /* ── i18n ─────────────────────────────────────────────── */
  t(key) { return this.i18n[this.currentLang][key] || this.i18n['en'][key] || key; }

  _switchLang(lang) {
    this.currentLang = lang;
    this.langText.textContent = lang === 'en' ? 'English' : '中文';
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = this.t('title');

    document.querySelectorAll('[data-i18n]').forEach(el => {
      const v = this.t(el.dataset.i18n);
      if (typeof v === 'string') {
        // footer 等允许含 HTML 的 key 用 innerHTML，其余保持 textContent
        if (el.dataset.i18n === 'footer_text') el.innerHTML = v;
        else el.textContent = v;
      }
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const v = this.t(el.dataset.i18nPlaceholder);
      if (typeof v === 'string') el.placeholder = v;
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      const v = this.t(el.dataset.i18nTitle);
      if (typeof v === 'string') el.title = v;
    });
  }

  /* ── Settings persistence ─────────────────────────────── */
  _saveSettings() {
    this._recordApiHistory(); // 双字段齐全才入历史（内部有守卫+去重），覆盖所有保存时机
    const s = {
      baseUrl:  this.modelBaseUrl.value,
      apiKey:   this.apiKeyInput.value,
      model:    this.modelSelect.value,
      summaryLang: this.summaryLangSel.value,
      oralCleanup: this.oralCleanupInput.checked,
      spkNum:   this.spkNumInput.value,
      uiLang:   this.currentLang,
    };
    try { localStorage.setItem('vt_settings', JSON.stringify(s)); } catch (_) {}
  }

  _loadSettings() {
    try {
      const raw = localStorage.getItem('vt_settings');
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.baseUrl)     this.modelBaseUrl.value = s.baseUrl;
      if (s.apiKey)      this.apiKeyInput.value  = s.apiKey;
      if (s.summaryLang) this.summaryLangSel.value = s.summaryLang;
      this.oralCleanupInput.checked = s.oralCleanup === true;
      if (s.spkNum) this.spkNumInput.value = s.spkNum;
      if (s.uiLang) this._savedUiLang = s.uiLang;
      // Model options will be restored after fetching
      this._savedModel = s.model || '';

      // Auto-open settings if credentials were saved
      if (s.baseUrl || s.apiKey) {
        this.settingsBody.classList.add('open');
        this.settingsToggle.classList.add('open');
        // Attempt to re-fetch model list silently
        if (s.baseUrl && s.apiKey) {
          setTimeout(() => this._fetchModels(true), 400);
        }
      }
    } catch (_) {}
  }

  /* ── API 配置历史（地址+密钥下拉复用）─────────────────── */
  _loadApiHistory() {
    try {
      const raw = localStorage.getItem('vt_api_history');
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr.filter(e => e && e.baseUrl && e.apiKey) : [];
    } catch (_) { return []; }
  }

  _saveApiHistory() {
    try { localStorage.setItem('vt_api_history', JSON.stringify(this._apiHistory.slice(0, 10))); } catch (_) {}
  }

  _recordApiHistory() {
    const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const apiKey  = this.apiKeyInput.value.trim();
    if (!baseUrl || !apiKey) return;
    this._apiHistory = this._apiHistory || this._loadApiHistory();
    const existing = this._apiHistory.findIndex(e => e.baseUrl === baseUrl && e.apiKey === apiKey);
    const entry = {
      baseUrl, apiKey,
      model: this.modelSelect.value || '',
      ts: new Date().toISOString().slice(0, 16).replace('T', ' '),
    };
    if (existing >= 0) this._apiHistory.splice(existing, 1); // 去重：老条目提到最前并刷新时间
    this._apiHistory.unshift(entry);
    this._apiHistory = this._apiHistory.slice(0, 10);
    this._saveApiHistory();
  }

  _maskKey(key) {
    return '••••••' + String(key || '').slice(-4);
  }

  _showApiHistory() {
    this._apiHistory = this._apiHistory || this._loadApiHistory();
    if (!this._apiHistory.length) { this._hideApiHistory(); return; }

    const curBase = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const curKey  = this.apiKeyInput.value.trim();
    const items = this._apiHistory.map((e, i) => {
      const active = e.baseUrl === curBase && e.apiKey === curKey;
      const meta = [this._maskKey(e.apiKey), e.model, e.ts].filter(Boolean).join(' · ');
      return `
        <div class="api-hist-item${active ? ' active' : ''}" data-idx="${i}">
          <div class="api-hist-main">
            <div class="api-hist-url">${this._escAttr(e.baseUrl)}</div>
            <div class="api-hist-meta">${this._escAttr(meta)}</div>
          </div>
          <button type="button" class="api-hist-del" data-del="${i}" title="${this.t('api_history_delete')}">
            <i class="fas fa-times"></i>
          </button>
        </div>`;
    }).join('');

    this.apiHistoryMenu.innerHTML = `
      <div class="api-hist-head">${this.t('api_history_title')}</div>
      ${items}`;
    this.apiHistoryMenu.hidden = false;

    this.apiHistoryMenu.querySelectorAll('.api-hist-item').forEach(el => {
      el.addEventListener('click', (ev) => {
        if (ev.target.closest('.api-hist-del')) return; // 删除按钮单独处理
        this._applyApiHistory(this._apiHistory[Number(el.dataset.idx)]);
      });
    });
    this.apiHistoryMenu.querySelectorAll('.api-hist-del').forEach(btn => {
      btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this._apiHistory.splice(Number(btn.dataset.del), 1);
        this._saveApiHistory();
        this._showApiHistory(); // 重渲染；空了会自动隐藏
      });
    });
  }

  _hideApiHistory() {
    if (this.apiHistoryMenu) { this.apiHistoryMenu.hidden = true; this.apiHistoryMenu.innerHTML = ''; }
  }

  _applyApiHistory(entry) {
    if (!entry) return;
    this.modelBaseUrl.value = entry.baseUrl;
    this.apiKeyInput.value  = entry.apiKey;
    this._savedModel = entry.model || '';
    this._hideApiHistory();
    this._saveSettings();
    // 派发 input 触发既有「双字段已填 → 防抖自动拉模型」逻辑，模型选中项由 _savedModel 恢复
    this.modelBaseUrl.dispatchEvent(new Event('input', { bubbles: true }));
    this.apiKeyInput.dispatchEvent(new Event('input', { bubbles: true }));
    this._showToast(this.t('api_history_applied'));
  }

  /* ── Fetch models ─────────────────────────────────────── */
  async _fetchModels(silent = false) {
    const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const apiKey  = this.apiKeyInput.value.trim();

    if (!baseUrl || !apiKey) {
      if (!silent) this._setFetchStatus('err', this.t('api_key') + ' & URL required');
      return;
    }

    this.fetchModelsBtn.disabled = true;
    this.fetchIcon.className = 'fas fa-spinner fa-spin';
    if (!silent) this._setFetchStatus('', this.t('fetching_models'));

    try {
      const fd = new FormData();
      fd.append('base_url', baseUrl);
      fd.append('api_key',  apiKey);

      const resp = await fetch(`${this.apiBase}/models`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const models = data.data || data.models || [];

      // Rebuild select options
      this.modelSelect.innerHTML = `<option value="">${this.t('model_default')}</option>`;
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name || m.id;
        this.modelSelect.appendChild(opt);
      });

      // Restore previously selected model
      if (this._savedModel) {
        this.modelSelect.value = this._savedModel;
        this._savedModel = '';
      }

      this._setFetchStatus('ok', typeof this.t('models_loaded') === 'function'
        ? this.t('models_loaded')(models.length)
        : `${models.length} models`);
      this._recordApiHistory(); // 模型拉取成功 = 这套配置验证可用，记入历史

    } catch (e) {
      console.warn('Model fetch error:', e);
      this._setFetchStatus('err', this.t('models_error') + ': ' + e.message);
    } finally {
      this.fetchModelsBtn.disabled = false;
      this.fetchIcon.className = 'fas fa-sync-alt';
    }
  }

  _setFetchStatus(cls, msg) {
    this.fetchStatus.className = 'fetch-status' + (cls ? ` ${cls}` : '');
    this.fetchStatus.textContent = msg;
  }

  /* ── Transcription ────────────────────────────────────── */
  async _startTranscription() {
    if (this.submitBtn.disabled) return;

    const url     = this.videoUrlInput.value.trim();
    const sumLang = this.summaryLangSel.value;

    if (!url) { this._showError(this.t('error_invalid_url')); return; }

    this._setLoading(true);
    this._hideError();
    this._showProgress();

    try {
      const fd = new FormData();
      fd.append('url',              url);
      fd.append('summary_language', sumLang);
      fd.append('oral_cleanup',     this.oralCleanupInput.checked ? 'true' : 'false');
      if (this.spkNumInput.value.trim()) fd.append('spk_num', this.spkNumInput.value.trim());

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Request failed');
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  async _startFileUpload(file) {
    if (this.submitBtn.disabled) return;

    const parts = (file.name || '').split('.');
    const ext = parts.length > 1 ? ('.' + parts.pop().toLowerCase()) : '';
    if (!this._allowedUploadExts.has(ext)) {
      this._showError(this.t('error_upload_type'));
      return;
    }
    if (!file.size) {
      this._showError(this.t('error_upload_empty'));
      return;
    }
    const maxB = this.uploadMaxMb * 1024 * 1024;
    if (file.size > maxB) {
      this._showError(this.t('error_upload_size')(this.uploadMaxMb));
      return;
    }

    this._setLoading(true);
    this._hideError();
    this._showProgress();

    const sumLang = this.summaryLangSel.value;
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      fd.append('summary_language', sumLang);
      fd.append('oral_cleanup',     this.oralCleanupInput.checked ? 'true' : 'false');
      if (this.spkNumInput.value.trim()) fd.append('spk_num', this.spkNumInput.value.trim());

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        const d = err.detail;
        const msg = typeof d === 'string'
          ? d
          : (Array.isArray(d) && d[0] && (d[0].msg || d[0].message))
            || `HTTP ${resp.status}`;
        throw new Error(msg);
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  /* ── SSE ──────────────────────────────────────────────── */
  _startSSE() {
    if (!this.currentTaskId) return;
    this.eventSource = new EventSource(`${this.apiBase}/task-stream/${this.currentTaskId}`);

    this.eventSource.onmessage = (ev) => {
      try {
        const task = JSON.parse(ev.data);
        if (task.type === 'heartbeat') return;

        this._updateProgress(task.progress, task.message, true);

        if (task.status === 'completed') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language, task.timeline, task.completed_at, task.total_duration, task.speaker_names);
        } else if (task.status === 'error') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showError(task.error || 'Processing error');
        }
      } catch (_) {}
    };

    this.eventSource.onerror = async () => {
      this._stopSSE();
      try {
        if (this.currentTaskId) {
          const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
          if (r.ok) {
            const task = await r.json();
            if (task?.status === 'completed') {
              this._stopSP(); this._setLoading(false); this._hideProgress();
              this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language, task.timeline, task.completed_at, task.total_duration, task.speaker_names);
              return;
            }
          }
        }
      } catch (_) {}
      this._showError(this.t('error_processing_failed') + 'SSE disconnected');
      this._setLoading(false);
    };
  }

  _stopSSE() {
    if (this.eventSource) { this.eventSource.close(); this.eventSource = null; }
  }

  /* ── Progress ─────────────────────────────────────────── */
  _updateProgress(pct, msg, fromServer = false) {
    if (fromServer) {
      this._stopSP();
      this.sp.lastServer = pct;
      this.sp.current    = pct;
      this._renderProgress(pct, msg);
      this._updateStage(pct, msg);
      this._startSP();
    } else {
      this._renderProgress(pct, msg);
    }
  }

  _updateStage(pct, msg) {
    const m = (msg || '').toLowerCase();

    // ── 字幕路径（快速）──────────────────────────────────────
    if (m.includes('获取成功') || m.includes('subtitle found') || m.includes('字幕获取')) {
      this.sp.stage = 'subtitle_found';
      this.sp.target = 55;
      this._setModeBadge('subtitle');
    }
    // ── 无字幕 → 音频下载路径（慢）────────────────────────────
    else if (m.includes('未找到字幕') || m.includes('no subtitle') || m.includes('下载视频音频') || m.includes('下载音频')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('读取文本') || (m.includes('read') && m.includes('text'))) {
      this.sp.stage = 'parsing';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('转换音频') || m.includes('准备转录')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('上传') || m.includes('upload')) {
      this.sp.stage = 'preparing';
      this.sp.target = 40;
    }
    // ── 通用字幕检测中 ─────────────────────────────────────────
    else if (m.includes('检测') && (m.includes('字幕') || m.includes('subtitle'))) {
      this.sp.stage = 'subtitle';
      this.sp.target = 40;
    }
    // ── 其他阶段 ───────────────────────────────────────────────
    else if (m.includes('解析') || m.includes('pars'))                     { this.sp.stage = 'parsing';       this.sp.target = 60; }
    else if (m.includes('下载') || m.includes('download'))                 { this.sp.stage = 'downloading';   this.sp.target = 60; }
    else if (m.includes('转录') || m.includes('transcrib') || m.includes('whisper')) { this.sp.stage = 'transcribing';  this.sp.target = 80; }
    else if (m.includes('优化') || m.includes('optimiz'))                  { this.sp.stage = 'optimizing';    this.sp.target = 90; }
    else if (m.includes('清理') || m.includes('clean'))                    { this.sp.stage = 'cleaning';      this.sp.target = 92; }
    else if (m.includes('摘要') || m.includes('summary'))                  { this.sp.stage = 'summarizing';   this.sp.target = 95; }
    else if (m.includes('完成') || m.includes('complet'))                  { this.sp.stage = 'completed';     this.sp.target = 100; }

    if (pct >= this.sp.target) this.sp.target = Math.min(pct + 8, 99);
  }

  _setModeBadge(mode) {
    if (!this.modeBadge) return;
    if (mode === 'subtitle') {
      this.modeBadge.textContent  = this.t('mode_subtitle');
      this.modeBadge.className    = 'mode-badge subtitle';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.add('subtitle-mode');
    } else if (mode === 'whisper') {
      this.modeBadge.textContent  = this.t('mode_whisper');
      this.modeBadge.className    = 'mode-badge whisper';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
    }
  }

  _initSP() {
    this.sp.enabled = false; this.sp.current = 0; this.sp.target = 15;
    this.sp.lastServer = 0;  this.sp.startTime = Date.now(); this.sp.stage = 'preparing';
  }
  _startSP() {
    if (this.sp.interval) clearInterval(this.sp.interval);
    this.sp.enabled   = true;
    this.sp.startTime = this.sp.startTime || Date.now();
    this.sp.interval  = setInterval(() => this._tickSP(), 500);
  }
  _stopSP() {
    if (this.sp.interval) { clearInterval(this.sp.interval); this.sp.interval = null; }
    this.sp.enabled = false;
  }
  _tickSP() {
    if (!this.sp.enabled || this.sp.current >= this.sp.target) return;
    const speeds = { subtitle: .5, parsing: .3, downloading: .18, transcribing: .14, optimizing: .22, cleaning: .2, summarizing: .28 };
    let inc = speeds[this.sp.stage] || .2;
    const remaining = this.sp.target - this.sp.current;
    if (remaining < 5) inc *= .3;
    const next = Math.min(this.sp.current + inc, this.sp.target);
    if (next > this.sp.current) {
      this.sp.current = next;
      this._renderProgress(next, this._stageMsg());
    }
  }
  _stageMsg() {
    const map = {
      subtitle:       this.t('detecting_subtitles'),
      subtitle_found: this.t('subtitle_found'),
      downloading:    this.t('downloading_video'),
      parsing:        this.t('parsing_video'),
      transcribing:   this.t('transcribing_audio'),
      optimizing:     this.t('optimizing_transcript'),
      cleaning:       this.t('cleaning_oral'),
      summarizing:    this.t('generating_summary'),
      completed:      this.t('completed'),
    };
    return map[this.sp.stage] || this.t('processing');
  }

  _renderProgress(pct, msg) {
    const p = Math.round(pct * 10) / 10;
    this.progressStatus.textContent = `${p}%`;
    this.progressFill.style.width   = `${p}%`;

    // Translate common server messages — more specific checks first
    const m = (msg || '').toLowerCase();
    let label = msg;
    // ── Subtitle path ──────────────────────────────────────────
    if      (m.includes('转录缓存') || m.includes('transcript cache'))    label = this.t('cache_hit_raw');
    else if (m.includes('音频缓存') || m.includes('audio cache'))        label = this.t('cache_hit_audio');
    else if (m.includes('获取成功') || m.includes('subtitle found'))        label = this.t('subtitle_found');
    else if (m.includes('未找到字幕') || m.includes('no subtitle'))         label = this.t('no_subtitle');
    else if (m.includes('检测') && (m.includes('字幕') || m.includes('subtitle'))) label = this.t('detecting_subtitles');
    // ── Audio / Whisper path ────────────────────────────────────
    else if (m.includes('下载') || m.includes('download'))  label = this.t('downloading_video');
    else if (m.includes('解析') || m.includes('pars'))      label = this.t('parsing_video');
    else if (m.includes('转录') || m.includes('transcrib')) label = this.t('transcribing_audio');
    else if (m.includes('清理') || m.includes('clean'))    label = this.t('cleaning_oral');
    else if (m.includes('优化') || m.includes('optimiz'))   label = this.t('optimizing_transcript');
    else if (m.includes('摘要') || m.includes('summary'))   label = this.t('generating_summary');
    else if (m.includes('完成') || m.includes('complet'))   label = this.t('completed');
    else if (m.includes('准备') || m.includes('prepar'))    label = this.t('preparing');

    this.progressMessage.textContent = label;
  }

  _showProgress() {
    this.emptyState.style.display    = 'none';
    this.resultsPanel.classList.remove('show');
    this.progressPanel.classList.add('show');
    // Reset mode badge & progress bar color for new task
    if (this.modeBadge) { this.modeBadge.style.display = 'none'; this.modeBadge.className = 'mode-badge'; }
    if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
  }
  _hideProgress() { this.progressPanel.classList.remove('show'); }

  /* ── Results ──────────────────────────────────────────── */
  /** 与后端 Translator.normalize_lang_code 对齐，用于 Tab 展示判断 */
  _normLangTab(code) {
    if (!code) return '';
    const c = String(code).toLowerCase().trim();
    if (c.startsWith('zh')) return 'zh';
    if (c.length >= 2) return c.slice(0, 2);
    return c;
  }

  _showResults(script, summary, videoTitle, translation, detectedLang, summaryLang, timeline, completedAt, totalDuration, speakerNames) {
    this.scriptContent.innerHTML  = script    ? marked.parse(script)      : '';
    this.summaryContent.innerHTML = summary   ? marked.parse(summary)     : '';

    const d = this._normLangTab(detectedLang);
    const s = this._normLangTab(summaryLang);
    const showTranslation = Boolean(translation) && d && s && d !== s;
    if (showTranslation) {
      this.translationContent.innerHTML = marked.parse(translation);
      this.translationTabBtn.style.display  = 'inline-block';
      this.dlTranslation.style.display      = 'inline-flex';
    } else {
      this.translationTabBtn.style.display  = 'none';
      this.dlTranslation.style.display      = 'none';
    }

    this._renderTaskMeta(timeline, completedAt, totalDuration);
    this._renderSpeakerPanel(script, speakerNames);
    this.resultsPanel.classList.add('show');
    this._switchTab('script');
    this.resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  _hideResults() { this.resultsPanel.classList.remove('show'); }

  /* ── 说话人命名面板：文稿含「说话人A：」标签时出现，映射真名后即时重渲染 ── */
  _renderSpeakerPanel(script, applied) {
    let panel = document.getElementById('speakerPanel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'speakerPanel';
      panel.className = 'speaker-panel';
      const tabBar = this.resultsPanel.querySelector('.tab-bar');
      this.resultsPanel.insertBefore(panel, tabBar);
    }

    const labels = [...new Set([...String(script || '').matchAll(/说话人([A-Za-z0-9]+)：/g)].map(m => m[1]))];
    const known = [...new Set([...labels, ...Object.keys(applied || {})])].sort();
    if (!known.length) { panel.style.display = 'none'; panel.innerHTML = ''; return; }

    const esc = (v) => String(v || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
    const rows = known.map(key => `
      <div class="spk-row">
        <span class="spk-row-label">${esc(`说话人${key}：`)}</span>
        <input type="text" class="spk-name-input" data-spk="${esc(key)}" maxlength="30" value="${esc((applied || {})[key] || '')}">
      </div>`).join('');

    panel.innerHTML = `
      <div class="spk-head">
        <span>🏷 ${this.t('speaker_rename_title')}</span>
        <button class="spk-apply-btn" id="speakerApplyBtn">${this.t('speaker_apply')}</button>
      </div>
      <div class="spk-hint">${this.t('speaker_rename_hint')}</div>
      <div class="spk-rows">${rows}</div>`;
    panel.style.display = '';
    panel.querySelector('#speakerApplyBtn').addEventListener('click', () => this._applySpeakerNames());
  }

  async _applySpeakerNames() {
    if (!this.currentTaskId) return;
    const mapping = {};
    document.querySelectorAll('#speakerPanel .spk-name-input').forEach(inp => {
      const v = inp.value.trim().replace(/[:：]/g, '');
      if (v) mapping[inp.dataset.spk] = v;
    });
    try {
      const resp = await fetch(`${this.apiBase}/task/${this.currentTaskId}/speakers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speakers: mapping }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const task = await resp.json();
      this._showResults(
        task.script, task.summary, task.video_title, task.translation,
        task.detected_language, task.summary_language, task.timeline,
        task.completed_at, task.total_duration, task.speaker_names,
      );
    } catch (err) {
      this._showError(this.t('speaker_apply_failed') + ': ' + err.message);
    }
  }

  /* ── Task meta（完成时间/耗时展示）── */
  _renderTaskMeta(timeline, completedAt, totalDuration) {
    let meta = document.getElementById('taskMeta');
    if (!meta) {
      meta = document.createElement('div');
      meta.id = 'taskMeta';
      meta.className = 'task-meta';
      this.resultsPanel.insertBefore(meta, this.resultsPanel.firstChild);
    }
    const parts = [];
    if (totalDuration) parts.push(`<span class="tm-total">⏱ <b>${this.t('total_time')} ${this._fmtDuration(totalDuration)}</b></span>`);
    if (timeline && typeof timeline === 'object') {
      const order = ['audio_download', 'transcribe', 'optimize', 'oral_cleanup', 'summarize', 'translate'];
      const segs = order.filter(s => timeline[s]).map(s => `${this._stageLabel(s)} ${this._fmtDuration(timeline[s])}`);
      if (segs.length) parts.push(`<span>${segs.join(' · ')}</span>`);
    }
    if (completedAt) parts.push(`<span class="tm-done">✓ ${this.t('finished_at')} ${this._fmtTime(completedAt)}</span>`);
    meta.innerHTML = parts.join('<br>');
  }
  _fmtDuration(sec) {
    sec = Math.round(Number(sec) || 0);
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const u = { h: this.t('dur_h'), m: this.t('dur_m'), s: this.t('dur_s') };
    if (h) return s ? `${h}${u.h}${m}${u.m}${s}${u.s}` : `${h}${u.h}${m}${u.m}`;
    if (m) return s ? `${m}${u.m}${s}${u.s}` : `${m}${u.m}`;
    return `${s}${u.s}`;
  }
  _fmtTime(iso) {
    try { return new Date(iso).toLocaleTimeString('zh-CN', { hour12: false }); }
    catch { return String(iso); }
  }
  _stageLabel(s) {
    return {
      subtitle_fetch: this.t('stage_subtitle'),
      audio_download: this.t('stage_download'),
      transcribe:     this.t('stage_transcribe'),
      optimize:       this.t('stage_optimize'),
      oral_cleanup:   this.t('stage_cleanup'),
      summarize:      this.t('stage_summarize'),
      translate:      this.t('stage_translate'),
    }[s] || s;
  }

  /* ── Tabs ─────────────────────────────────────────────── */
  _switchTab(name) {
    this.tabBtns.forEach(b  => b.classList.toggle('active',  b.dataset.tab === name));
    this.tabPanes.forEach(p => p.classList.toggle('active', p.id === `${name}Tab`));
  }

  /* ── Download ─────────────────────────────────────────── */
  async _downloadFile(type) {
    if (!this.currentTaskId) { this._showError(this.t('error_no_download')); return; }
    try {
      const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
      if (!r.ok) throw new Error('Failed to get task status');
      const task = await r.json();

      let filename;
      if      (type === 'script')      filename = task.script_path      ? String(task.script_path).split(/[\\/]/).pop()      : `transcript_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else if (type === 'summary')     filename = task.summary_path     ? String(task.summary_path).split(/[\\/]/).pop()     : `summary_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else if (type === 'translation') filename = task.translation_path ? String(task.translation_path).split(/[\\/]/).pop() : `translation_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else throw new Error('Unknown type');

      const a = document.createElement('a');
      a.href = `${this.apiBase}/download/${encodeURIComponent(filename)}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      this._showError(this.t('error_download_failed') + e.message);
    }
  }

  /* ── Batch editor ─────────────────────────────────────── */
  _escAttr(v) {
    return String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  _onUrlPaste(e) {
    const text = (e.clipboardData && e.clipboardData.getData('text')) || '';
    const urls = [...new Set((text.match(/https?:\/\/[^\s"'<>）)]+/g) || []).map(u => u.trim()))];
    if (urls.length < 2) return; // 单链接：走默认粘贴
    e.preventDefault();
    this._openBatchEditor(urls.slice(0, 10));
    if (urls.length > 10) this._showBatchNotice(this.t('batch_truncated_notice'));
  }

  _openBatchEditor(urls) {
    this.batchOpen = true;
    this.batchToggle.classList.add('active');
    this.singleUrlWrap.style.display = 'none';
    this.videoUrlInput.removeAttribute('required'); // 隐藏的 required 输入会阻塞表单提交
    if (this.settingsRow) this.settingsRow.classList.add('batch-on'); // 隐藏全局单任务设置（行内已有）
    this.batchEditor.hidden = false;
    this._clearBatchRows();
    (urls.length ? urls : ['']).forEach(u => this._addBatchRow(u));
    this._hideBatchNotice();
    this.batchEditor.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  _closeBatchEditor() {
    this.batchOpen = false;
    this.batchToggle.classList.remove('active');
    this.singleUrlWrap.style.display = '';
    this.videoUrlInput.setAttribute('required', '');
    if (this.settingsRow) this.settingsRow.classList.remove('batch-on');
    this.batchEditor.hidden = true;
    this._hideBatchNotice();
    this._renderSubmitBtn();
  }

  _addBatchRow(url) {
    if (this.batchRowsEl.querySelectorAll('.batch-row').length >= 10) {
      this._showBatchNotice(this.t('batch_max_hint'));
      return;
    }
    const row = document.createElement('div');
    row.className = 'batch-row';
    const langOptions = [...this.summaryLangSel.options]
      .map(o => `<option value="${this._escAttr(o.value)}">${o.textContent}</option>`).join('');
    row.innerHTML = `
      <span class="batch-idx"></span>
      <input type="url" class="batch-url-input" spellcheck="false"
             placeholder="${this.t('batch_url_placeholder')}" value="${this._escAttr(url || '')}">
      <select class="batch-lang">${langOptions}</select>
      <label class="batch-cleanup" data-i18n-title="oral_cleanup_hint" title="Remove filler words & smooth sentences (not verbatim)">
        <input type="checkbox" class="switch-input batch-cleanup-input"${this.oralCleanupInput.checked ? ' checked' : ''}>
        <span class="switch-ui"></span>
      </label>
      <input type="number" class="batch-spk" min="1" max="30" step="1" data-i18n-placeholder="spk_num_auto"
             placeholder="Auto" title="${this.t('spk_num')}" value="${this._escAttr(this.spkNumInput.value)}">
      <button type="button" class="batch-remove" title="${this.t('batch_remove')}"><i class="fas fa-times"></i></button>
    `;
    row.querySelector('.batch-lang').value = this.summaryLangSel.value;
    this.batchRowsEl.appendChild(row);
    this._renumberBatchRows();
    this._validateBatch();
  }

  _renumberBatchRows() {
    this.batchRowsEl.querySelectorAll('.batch-row').forEach((row, i) => {
      row.querySelector('.batch-idx').textContent = `${i + 1}.`;
    });
  }

  _clearBatchRows() {
    this.batchRowsEl.innerHTML = '';
    this._validateBatch();
  }

  _batchRowsData() {
    return [...this.batchRowsEl.querySelectorAll('.batch-row')].map(row => ({
      row,
      url: row.querySelector('.batch-url-input').value.trim(),
      lang: row.querySelector('.batch-lang').value,
      cleanup: row.querySelector('.batch-cleanup-input').checked,
      spk: row.querySelector('.batch-spk').value.trim(),
    }));
  }

  _validateBatch() {
    const rows = this._batchRowsData();
    const seen = new Map();
    rows.forEach(r => { if (r.url) seen.set(r.url, (seen.get(r.url) || 0) + 1); });
    let allValid = rows.length > 0;
    rows.forEach(r => {
      const valid = /^https?:\/\/\S+$/.test(r.url);
      const dup = !!r.url && seen.get(r.url) > 1;
      r.row.classList.toggle('invalid', !valid);
      r.row.classList.toggle('dup', valid && dup);
      let hint = r.row.querySelector('.batch-row-hint');
      const msg = !valid ? this.t('batch_invalid_url') : (dup ? this.t('batch_duplicate') : '');
      if (msg) {
        if (!hint) { hint = document.createElement('div'); hint.className = 'batch-row-hint'; r.row.appendChild(hint); }
        hint.textContent = msg;
      } else if (hint) {
        hint.remove();
      }
      if (!valid) allValid = false;
    });
    this._renderSubmitBtn();
  }

  _showBatchNotice(msg, cls) {
    this.batchNotice.textContent = msg;
    this.batchNotice.className = 'batch-notice' + (cls ? ` ${cls}` : '');
    this.batchNotice.hidden = false;
  }
  _hideBatchNotice() { this.batchNotice.hidden = true; this.batchNotice.textContent = ''; }

  async _submitBatch() {
    if (this.submitBtn.disabled) return;
    const rows = this._batchRowsData();
    const valid = rows.filter(r => /^https?:\/\/\S+$/.test(r.url));
    if (!valid.length) { this._showError(this.t('batch_invalid_url')); return; }

    const apiKey  = this.apiKeyInput.value.trim();
    const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const modelId = this.modelSelect.value;
    const items = valid.map(r => ({
      url: r.url,
      summary_language: r.lang,
      oral_cleanup: r.cleanup,
      spk_num: r.spk,
      api_key: apiKey,
      model_base_url: baseUrl,
      model_id: modelId,
    }));

    this._setLoading(true);
    this._hideError();
    try {
      const resp = await fetch(`${this.apiBase}/process-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();

      if (data.skipped && data.skipped.length) {
        this._showError(this.t('batch_skipped')
          + data.skipped.map(s => `${s.url}（${s.reason}）`).join('；'));
      }

      this._closeBatchEditor();
      this.videoUrlInput.value = '';
      this._saveSettings();
      this._loadTaskList();

      const tasks = data.tasks || [];
      if (tasks.length === 1) {
        // 单任务：沿用既有单任务体验（进度面板 + SSE）
        this.currentTaskId = tasks[0].task_id;
        this._showProgress();
        this._initSP();
        this._updateProgress(5, this.t('preparing'), true);
        this._startSSE();
      } else {
        // 多任务：交给任务列表展示
        this._setLoading(false);
      }
    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
    }
  }

  /* ── Task list panel ──────────────────────────────────── */
  _startTaskListPolling() {
    this._loadTaskList();
    this._taskPollTimer = setInterval(() => {
      if (document.visibilityState === 'visible') this._loadTaskList();
    }, 4000);
  }

  _stopTaskListPolling() {
    if (this._taskPollTimer) { clearInterval(this._taskPollTimer); this._taskPollTimer = null; }
  }

  async _loadTaskList() {
    try {
      const resp = await fetch(`${this.apiBase}/tasks`);
      if (!resp.ok) return;
      const data = await resp.json();
      this._lastTasks = data.tasks || [];
      this._renderTaskList(this._lastTasks);
    } catch (_) { /* 静默：下一轮轮询重试 */ }
  }

  _taskState(t) {
    const qs = t.queue_status;
    if (qs === 'queued') return 'queued';
    if (qs === 'running') return 'running';
    if (qs === 'interrupted') return 'interrupted';
    if (qs === 'cancelled') return 'cancelled';
    if (t.status === 'completed') return 'completed';
    if (t.status === 'error') return 'error';
    if (t.status === 'cancelled') return 'cancelled';
    if (t.status === 'processing') return 'running';
    return 'queued';
  }

  _isTerminal(t) {
    const st = this._taskState(t);
    return st === 'completed' || st === 'error' || st === 'cancelled' || st === 'interrupted';
  }

  _renderTaskList(tasks) {
    if (!tasks.length) {
      this.taskListPanel.hidden = true;
      this.taskListBody.innerHTML = '';
      return;
    }
    this.taskListPanel.hidden = false;

    const count = { total: tasks.length, done: 0, failed: 0, running: 0, queued: 0, interrupted: 0 };
    tasks.forEach(t => {
      const st = this._taskState(t);
      if (st === 'completed') count.done++;
      else if (st === 'error') count.failed++;
      else if (st === 'running') count.running++;
      else if (st === 'queued') count.queued++;
      else if (st === 'interrupted') count.interrupted++;
    });
    this.taskListStats.innerHTML = `
      <span class="tl-stat">${this.t('stat_total')} ${count.total}</span>
      <span class="tl-stat done">${this.t('stat_done')} ${count.done}</span>
      <span class="tl-stat failed">${this.t('stat_failed')} ${count.failed}</span>
      <span class="tl-stat running">${this.t('stat_running')} ${count.running}</span>
      <span class="tl-stat">${this.t('stat_queued')} ${count.queued}</span>
      ${count.interrupted ? `<span class="tl-stat interrupted">${this.t('stat_interrupted')} ${count.interrupted}</span>` : ''}
    `;

    // 按连续 batch_id 分组
    const groups = [];
    let cur = null;
    tasks.forEach(t => {
      if (t.batch_id && cur && cur.batchId === t.batch_id) cur.tasks.push(t);
      else { cur = { batchId: t.batch_id || null, tasks: [t] }; groups.push(cur); }
    });

    this.taskListBody.innerHTML = '';
    groups.forEach(g => {
      if (g.batchId && g.tasks.length > 1) this.taskListBody.appendChild(this._renderGroupHead(g));
      g.tasks.forEach(t => this.taskListBody.appendChild(this._renderTaskRow(t)));
    });
  }

  _renderGroupHead(g) {
    const head = document.createElement('div');
    head.className = 'task-group-head';
    const title = document.createElement('span');
    title.className = 'task-group-title';
    title.textContent = this.t('task_batch_group')(g.tasks.length);
    head.appendChild(title);
    const allTerminal = g.tasks.every(t => this._isTerminal(t));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-merge';
    btn.innerHTML = `<i class="fas fa-object-group"></i> <span>${this.t('task_merge')}</span>`;
    btn.disabled = !allTerminal;
    if (!allTerminal) btn.title = this.t('task_merge_disabled_hint');
    btn.addEventListener('click', () => this._mergeBatch(g.tasks.map(t => t.id)));
    head.appendChild(btn);
    return head;
  }

  _renderTaskRow(t) {
    const st = this._taskState(t);
    const row = document.createElement('div');
    row.className = 'task-row' + (st === 'running' ? ' running' : '');

    const main = document.createElement('div');
    main.className = 'task-row-main';

    const dot = document.createElement('span');
    dot.className = 'task-dot';
    main.appendChild(dot);

    const chip = document.createElement('span');
    chip.className = `task-chip ${st}`;
    chip.textContent = this.t('chip_' + st) || st;
    main.appendChild(chip);

    const title = document.createElement('span');
    title.className = 'task-title';
    const tText = t.video_title || t.url || t.id;
    title.textContent = tText;
    title.title = tText;
    main.appendChild(title);
    row.appendChild(main);

    if (st === 'running' && typeof t.progress === 'number') {
      const prog = document.createElement('div');
      prog.className = 'task-progress';
      const fill = document.createElement('div');
      fill.className = 'task-progress-fill';
      fill.style.width = `${Math.max(0, Math.min(100, t.progress))}%`;
      prog.appendChild(fill);
      row.appendChild(prog);
    }

    const metaBits = [];
    if (t.summary_language) metaBits.push(`<span class="task-meta-chip">${this._escAttr(t.summary_language)}</span>`);
    if (t.oral_cleanup) metaBits.push(`<span class="task-meta-chip">${this.t('task_meta_cleanup')} ✓</span>`);
    if (t.spk_num) metaBits.push(`<span class="task-meta-chip">${this.t('task_meta_spk')(this._escAttr(t.spk_num))}</span>`);
    if (t.cache_hit) metaBits.push(`<span class="task-meta-chip">${this._escAttr(t.cache_hit)}</span>`);
    if (st === 'completed' && t.completed_at) {
      metaBits.push(`<span class="task-meta-chip">${this.t('finished_at')} ${this._escAttr(t.completed_at)}</span>`);
      if (t.total_duration) metaBits.push(`<span class="task-meta-chip">${this.t('total_time')} ${t.total_duration}s</span>`);
    }
    if (st === 'error' && t.error) {
      metaBits.push(`<span class="task-meta-chip" style="color:var(--red)">${this._escAttr(String(t.error).slice(0, 80))}</span>`);
    }
    if (metaBits.length) {
      const meta = document.createElement('div');
      meta.className = 'task-meta-line';
      meta.innerHTML = metaBits.join('');
      row.appendChild(meta);
    }

    const actions = document.createElement('div');
    actions.className = 'task-actions';
    const mk = (act, icon, label, extraCls) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'task-btn' + (extraCls ? ` ${extraCls}` : '');
      b.dataset.taskAction = act;
      b.dataset.taskId = t.id;
      b.innerHTML = `<i class="fas ${icon}"></i> <span>${label}</span>`;
      return b;
    };
    if (st === 'queued') {
      actions.appendChild(mk('cancel', 'fa-ban', this.t('task_cancel')));
    } else if (st === 'completed') {
      actions.appendChild(mk('view', 'fa-eye', this.t('task_view')));
      if (t.has_script) actions.appendChild(mk('dl-script', 'fa-download', this.t('task_dl_script')));
      if (t.has_summary) actions.appendChild(mk('dl-summary', 'fa-download', this.t('task_dl_summary')));
    } else if (st === 'error') {
      actions.appendChild(mk('requeue', 'fa-rotate-right', this.t('task_retry')));
      actions.appendChild(mk('view', 'fa-eye', this.t('task_view')));
    } else if (st === 'interrupted') {
      actions.appendChild(mk('requeue', 'fa-rotate-right', this.t('task_requeue')));
    }
    if (actions.children.length) row.appendChild(actions);
    return row;
  }

  async _viewTask(id) {
    try {
      const resp = await fetch(`${this.apiBase}/task-status/${id}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const task = await resp.json();
      this.currentTaskId = id;
      this._showResults(
        task.script, task.summary, task.video_title, task.translation,
        task.detected_language, task.summary_language, task.timeline,
        task.completed_at, task.total_duration, task.speaker_names,
      );
    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
    }
  }

  async _downloadTaskFile(id, type) {
    try {
      const resp = await fetch(`${this.apiBase}/task-status/${id}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const task = await resp.json();
      const path = type === 'script' ? task.script_path : task.summary_path;
      if (!path) { this._showError(this.t('error_no_download')); return; }
      const filename = String(path).split(/[\\/]/).pop();
      const a = document.createElement('a');
      a.href = `${this.apiBase}/download/${encodeURIComponent(filename)}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      this._showError(this.t('error_download_failed') + err.message);
    }
  }

  async _taskAction(id, action) {
    try {
      const resp = await fetch(`${this.apiBase}/task/${id}/${action}`, { method: 'POST' });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      this._loadTaskList();
    } catch (err) {
      this._showError(this.t('task_action_failed') + err.message);
    }
  }

  async _mergeBatch(taskIds) {
    if (!taskIds || !taskIds.length) return;
    try {
      const resp = await fetch(`${this.apiBase}/batch/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_ids: taskIds }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      if (data.download_url) {
        const a = document.createElement('a');
        a.href = data.download_url;
        a.download = data.filename || '';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
      this._showToast(this.t('task_merge_done'));
      if (data.failed && data.failed.length) {
        this._showError(this.t('batch_skipped')
          + data.failed.map(f => `${f.url}（${f.reason}）`).join('；'));
      }
    } catch (err) {
      this._showError(this.t('task_merge_failed') + err.message);
    }
  }

  _showToast(msg) {
    if (!this.toastEl) return;
    this.toastEl.textContent = msg;
    this.toastEl.classList.add('show');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => this.toastEl.classList.remove('show'), 3200);
  }

  /* ── UI helpers ───────────────────────────────────────── */
  _renderSubmitBtn() {
    if (this.batchOpen) {
      const n = this.batchRowsEl.querySelectorAll('.batch-row').length;
      const allValid = n > 0 && [...this.batchRowsEl.querySelectorAll('.batch-row')]
        .every(r => /^https?:\/\/\S+$/.test(r.querySelector('.batch-url-input').value.trim()));
      this.submitBtn.disabled = this._loading || !allValid;
      this.submitBtn.innerHTML = `<i class="fas fa-layer-group"></i> <span>${this.t('batch_submit')(n)}</span>`;
    } else {
      this.submitBtn.disabled = this._loading;
      this.submitBtn.innerHTML = this._loading
        ? `<span class="spinner"></span> ${this.t('processing')}`
        : `<i class="fas fa-arrow-right"></i> <span>${this.t('start_transcription')}</span>`;
    }
  }

  _setLoading(on) {
    this._loading = on;
    this._renderSubmitBtn();
    if (this.uploadPickBtn) this.uploadPickBtn.disabled = on;
    if (this.uploadZone) {
      this.uploadZone.style.pointerEvents = on ? 'none' : '';
      this.uploadZone.style.opacity = on ? '0.65' : '';
      this.uploadZone.tabIndex = on ? -1 : 0;
    }
    if (this.fileInput) this.fileInput.disabled = on;
  }

  _showError(msg) {
    this.errorMsg.textContent = msg;
    this.errorBanner.classList.add('show');
    this.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setTimeout(() => this._hideError(), 6000);
  }
  _hideError() { this.errorBanner.classList.remove('show'); }

  _debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }
}

/* ── Boot ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  window.vt = new VideoTranscriber();
});

window.addEventListener('beforeunload', () => {
  if (window.vt?.eventSource) window.vt._stopSSE();
  if (window.vt?._taskPollTimer) window.vt._stopTaskListPolling();
});
