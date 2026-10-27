/* ------------------------------------------------------------------------
 * Les Chroniques du CO
 *   1. Ambiances sonores synthétisées (Web Audio, aucun fichier externe)
 *   2. Séquence « Souviens-toi »
 *   3. Candidature « Qui sera Andrea & Kransvelt ? »
 * Le tirage vit dans assets/secrets.js (window.CO_CONFIG).
 * ---------------------------------------------------------------------- */
(function () {
  'use strict';

  /* =====================================================================
   * 1. Ambiances sonores
   * =================================================================== */
  (function ambiances() {
    var AC = null, master = null, timer = null, activeBtn = null;

    function ctx() {
      if (!AC) {
        var C = window.AudioContext || window.webkitAudioContext;
        if (!C) { return null; }
        AC = new C();
      }
      if (AC.state === 'suspended' && AC.resume) { AC.resume(); }
      return AC;
    }

    function noise(ac, dur) {
      var len = Math.max(1, Math.floor(ac.sampleRate * dur));
      var buf = ac.createBuffer(1, len, ac.sampleRate);
      var d = buf.getChannelData(0);
      for (var i = 0; i < len; i++) { d[i] = Math.random() * 2 - 1; }
      var src = ac.createBufferSource();
      src.buffer = buf;
      return src;
    }

    function tone(ac, t, freq, dur, type, vol, glide) {
      var o = ac.createOscillator(), g = ac.createGain();
      o.type = type || 'sine';
      o.frequency.setValueAtTime(freq, t);
      if (glide) { o.frequency.exponentialRampToValueAtTime(Math.max(20, glide), t + dur); }
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(vol, t + Math.min(0.03, dur * 0.2));
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g); g.connect(master);
      o.start(t); o.stop(t + dur + 0.05);
    }

    function hit(ac, t, dur, freqLo, freqHi, vol) {
      var src = noise(ac, dur), f = ac.createBiquadFilter(), g = ac.createGain();
      f.type = 'bandpass';
      f.frequency.setValueAtTime(freqHi, t);
      f.frequency.exponentialRampToValueAtTime(Math.max(60, freqLo), t + dur);
      f.Q.value = 1.1;
      g.gain.setValueAtTime(vol, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      src.connect(f); f.connect(g); g.connect(master);
      src.start(t); src.stop(t + dur + 0.02);
    }

    function bed(ac, t, dur, freq, q, vol, lfoRate, lfoDepth) {
      var src = noise(ac, dur), f = ac.createBiquadFilter(), g = ac.createGain();
      f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = q;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.linearRampToValueAtTime(vol, t + 0.5);
      g.gain.linearRampToValueAtTime(0.0001, t + dur);
      if (lfoRate) {
        var lfo = ac.createOscillator(), lg = ac.createGain();
        lfo.frequency.value = lfoRate; lg.gain.value = lfoDepth || 200;
        lfo.connect(lg); lg.connect(f.frequency);
        lfo.start(t); lfo.stop(t + dur);
      }
      src.connect(f); f.connect(g); g.connect(master);
      src.start(t); src.stop(t + dur + 0.05);
    }

    var AMB = {
      /* caves de Reims : bouchon, pétillement, bulles */
      reims: function (ac, t) {
        hit(ac, t, 0.12, 120, 2400, 0.55);
        tone(ac, t, 700, 0.14, 'sine', 0.25, 160);
        bed(ac, t + 0.1, 5.2, 5200, 0.7, 0.045);
        for (var i = 0; i < 46; i++) {
          tone(ac, t + 0.25 + Math.random() * 4.8, 1400 + Math.random() * 2200, 0.05, 'sine', 0.05);
        }
        return 5.6;
      },
      /* stade d'Eindhoven : foule, claps, le riff que tout le monde hurle */
      eindhoven: function (ac, t) {
        bed(ac, t, 6, 480, 0.5, 0.075, 0.3, 120);
        for (var i = 0; i < 10; i++) { hit(ac, t + 0.35 + i * 0.55, 0.07, 400, 3600, 0.16); }
        var riff = [[0, 82.41], [0.52, 82.41], [0.78, 98.0], [1.04, 82.41], [1.3, 73.42], [1.82, 65.41], [2.5, 61.74]];
        for (var p = 0; p < 2; p++) {
          for (var j = 0; j < riff.length; j++) {
            var last = j === riff.length - 1;
            tone(ac, t + 0.4 + p * 3.1 + riff[j][0], riff[j][1], last ? 0.8 : 0.3, 'sawtooth', 0.16);
            tone(ac, t + 0.4 + p * 3.1 + riff[j][0], riff[j][1] * 2, last ? 0.8 : 0.3, 'triangle', 0.07);
          }
        }
        return 6.4;
      },
      /* cantons : oompah, bancs et chopes */
      cantons: function (ac, t) {
        var step = 0.42, bass = [49, 73.42], chord = [146.83, 185, 220];
        for (var b = 0; b < 12; b++) {
          var tt = t + 0.2 + b * step;
          if (b % 2 === 0) {
            tone(ac, tt, bass[(b / 2) % 2], 0.3, 'triangle', 0.3);
          } else {
            for (var c = 0; c < chord.length; c++) { tone(ac, tt, chord[c], 0.18, 'square', 0.07); }
          }
        }
        bed(ac, t, 5.6, 700, 0.4, 0.05, 0.25, 90);
        hit(ac, t + 2.3, 0.18, 180, 900, 0.3);
        hit(ac, t + 2.48, 0.22, 150, 700, 0.26);
        return 5.6;
      },
      /* Ardenne : bourdon, boîte à musique, parquet qui craque */
      ardenne: function (ac, t) {
        tone(ac, t, 55, 6.2, 'sine', 0.2);
        tone(ac, t, 82.41, 6.2, 'sine', 0.08);
        var mel = [440, 523.25, 659.25, 880, 659.25, 523.25, 440, 415.3];
        for (var i = 0; i < mel.length; i++) {
          tone(ac, t + 0.6 + i * 0.62, mel[i], 0.55, 'triangle', 0.13);
        }
        hit(ac, t + 2.1, 0.5, 90, 400, 0.12);
        hit(ac, t + 4.6, 0.6, 70, 300, 0.14);
        return 6.4;
      },
      /* Le Touquet : vent, sable, mouettes */
      touquet: function (ac, t) {
        bed(ac, t, 6.4, 520, 0.6, 0.14, 0.18, 380);
        bed(ac, t + 0.3, 5.8, 180, 0.8, 0.09, 0.11, 120);
        [1.1, 2.6, 4.4].forEach(function (d) {
          tone(ac, t + d, 1800, 0.22, 'sine', 0.1, 900);
          tone(ac, t + d + 0.26, 1600, 0.18, 'sine', 0.08, 800);
        });
        return 6.6;
      },
      /* Dinant : coup de sifflet et fanfare */
      dinant: function (ac, t) {
        tone(ac, t, 2300, 0.42, 'square', 0.1, 2100);
        var up = [261.63, 329.63, 392, 523.25];
        for (var i = 0; i < up.length; i++) {
          tone(ac, t + 0.7 + i * 0.26, up[i], 0.3, 'sawtooth', 0.13);
          tone(ac, t + 0.7 + i * 0.26, up[i] * 2, 0.3, 'triangle', 0.05);
        }
        up.forEach(function (f) { tone(ac, t + 1.85, f, 2.4, 'sawtooth', 0.1); });
        hit(ac, t + 1.85, 0.5, 80, 700, 0.22);
        return 4.6;
      }
    };

    function stop() {
      if (timer) { clearTimeout(timer); timer = null; }
      if (master && AC) {
        try {
          master.gain.cancelScheduledValues(AC.currentTime);
          master.gain.setValueAtTime(master.gain.value || 0.0001, AC.currentTime);
          master.gain.exponentialRampToValueAtTime(0.0001, AC.currentTime + 0.12);
        } catch (e) { /* le navigateur a coupé avant nous */ }
      }
      master = null;
      if (activeBtn) {
        activeBtn.classList.remove('playing');
        activeBtn.textContent = "Écouter l'ambiance";
        activeBtn = null;
      }
    }

    function play(key, btn) {
      var ac = ctx();
      if (!ac || !AMB[key]) { return false; }
      stop();
      master = ac.createGain();
      master.gain.value = 0.9;
      master.connect(ac.destination);
      var dur;
      try { dur = AMB[key](ac, ac.currentTime + 0.06); } catch (e) { return false; }
      activeBtn = btn;
      btn.classList.add('playing');
      btn.textContent = 'Couper';
      timer = setTimeout(stop, dur * 1000);
      return true;
    }

    Array.prototype.forEach.call(document.querySelectorAll('[data-amb]'), function (btn) {
      btn.addEventListener('click', function () {
        if (activeBtn === btn) { stop(); return; }
        if (!play(btn.getAttribute('data-amb'), btn)) {
          btn.textContent = 'Son indisponible';
          btn.disabled = true;
        }
      });
    });
  })();

  /* =====================================================================
   * 2. Séquence « Souviens-toi »
   * =================================================================== */
  (function souviens() {
    var MEM = [
      "Souviens-toi d'une cave à Reims, et de la lenteur avec laquelle on en est remonté.",
      "Souviens-toi d'une énigme que personne n'a résolue, à l'heure précise où plus personne ne pouvait rien résoudre.",
      "Souviens-toi de deux organisateurs attablés avec l'air de ne pas vouloir être vus. Devant onze témoins.",
      "Souviens-toi de onze casques, trois musiques, et d'un slow dansé à trente centimètres d'une techno.",
      "Souviens-toi d'une trottinette électrique, et d'une route qui n'avait rien demandé.",
      "Souviens-toi du genou d'Andrea dans la sciure, et du oui de Claire devant deux cents inconnus.",
      "Souviens-toi que Pauline est entrée en civile sur ce terrain de paintball.",
      "Souviens-toi que tu as joué ton propre avenir, et que certains l'ont joué beaucoup trop bien.",
      "Souviens-toi de la boîte noire, et de ce qu'on a tous appris sur Lara ce soir-là.",
      "Souviens-toi d'un appareil photo qui n'avait rien demandé non plus, et qui a tout vu.",
      "Souviens-toi du vent du Touquet : la seule fois où le CO est allé plus vite que prévu.",
      "Souviens-toi de Mimi, la chanteuse de diamant. Elle ne se souvient pas de toi.",
      "Souviens-toi d'un drapeau cousu le matin et défendu l'après-midi comme une patrie millénaire.",
      "Souviens-toi du warpool. Non. Ne t'en souviens pas.",
      "Souviens-toi qu'il s'est écoulé neuf mois, et que quatre étaient la règle.",
      "Souviens-toi qu'Andrea n'a jamais rien organisé. Et que Kransvelt non plus."
    ];

    var stage = document.getElementById('souviens-stage');
    var go = document.getElementById('souviens-go');
    var skip = document.getElementById('souviens-skip');
    if (!stage || !go || !skip) { return; }

    var i = 0, t = null;
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function line(text, cls) {
      var el = document.createElement('p');
      el.className = 'memory' + (cls ? ' ' + cls : '');
      el.textContent = text;
      return el;
    }

    function step() {
      Array.prototype.forEach.call(stage.querySelectorAll('.memory'), function (el) { el.classList.add('dim'); });
      stage.appendChild(line(MEM[i], i === MEM.length - 1 ? 'last' : ''));
      while (stage.children.length > 3) { stage.removeChild(stage.firstChild); }
      i++;
      if (i < MEM.length) {
        t = setTimeout(step, 2300);
      } else {
        go.textContent = 'Revoir';
        go.disabled = false;
        skip.hidden = true;
        i = 0;
      }
    }

    function all() {
      if (t) { clearTimeout(t); t = null; }
      stage.innerHTML = '';
      MEM.forEach(function (m, k) { stage.appendChild(line(m, k === MEM.length - 1 ? 'last' : '')); });
      go.textContent = 'Rejouer';
      go.disabled = false;
      skip.hidden = true;
      i = 0;
    }

    go.addEventListener('click', function () {
      if (t) { clearTimeout(t); t = null; }
      stage.innerHTML = '';
      i = 0;
      if (reduced) { all(); return; }
      go.disabled = true;
      go.textContent = 'En cours…';
      skip.hidden = false;
      step();
    });
    skip.addEventListener('click', all);
  })();

  /* =====================================================================
   * 3. « Qui sera Andrea & Kransvelt ? »
   * =================================================================== */
  (function candidature() {
    var CFG = window.CO_CONFIG;

    var QUESTIONS = [
      { q: "Le groupe te demande le programme trois semaines avant le départ. Tu réponds :", o: [
        ["Un indice par semaine, pas un de plus.", 12],
        ["Rien. Le silence est une activité à part entière.", 10],
        ["Un fichier avec les horaires et les temps de trajet.", 2],
        ["Je transfère la question au Comité et je disparais.", 6]
      ]},
      { q: "Le budget, tu l'annonces :", o: [
        ["Dès le départ, à l'euro près.", 10],
        ["Juste après l'encaissement de l'acompte.", 6],
        ["Par tranches, pour amortir le choc.", 8],
        ["Jamais. On verra bien sur place.", 3]
      ]},
      { q: "On te propose de réintroduire le warpool.", o: [
        ["Non.", 14],
        ["Non, mais avec un règlement écrit cette fois.", 4],
        ["Oui, et en version aquatique.", 1],
        ["Je demande d'abord l'avis de Pauline.", 8]
      ]},
      { q: "Le gîte ne dispose que d'une salle de bain pour onze. Pour toi :", o: [
        ["Je vérifie le nombre de douches avant de réserver.", 12],
        ["C'est un puissant facteur de cohésion.", 7],
        ["J'affiche un tableau de rotation dans le couloir.", 9],
        ["Je n'avais jamais songé à regarder.", 2]
      ]},
      { q: "L'activité physique obligatoire, tu la calibres :", o: [
        ["Correctement, comme le char à voile.", 12],
        ["Légèrement au-dessus des capacités du groupe.", 9],
        ["Très au-dessus. C'est ainsi qu'on fabrique des légendes.", 5],
        ["Je laisse le hasard et la météo décider.", 3]
      ]},
      { q: "Lara perd la finale du jeu que tu avais prévu. Ton plan :", o: [
        ["Un second jeu, immédiatement, qu'elle peut gagner.", 11],
        ["Je truque discrètement le premier.", 8],
        ["Je ne prévois aucune finale, justement.", 6],
        ["Aucun plan. J'accepte mon sort.", 4]
      ]},
      { q: "Combien de groupes de discussion secrets es-tu prêt à créer ?", o: [
        ["Zéro. Tout passe par la boîte anonyme.", 12],
        ["J'en ai déjà créé un en lisant la question.", 10],
        ["Un, pour les décisions sérieuses.", 8],
        ["Trois, dont un pour parler des deux autres.", 6]
      ]},
      { q: "Neuf mois se sont écoulés depuis la dernière édition. Pour toi, c'est :", o: [
        ["Un scandale, et quelqu'un doit en répondre.", 12],
        ["Je viens seulement de m'en rendre compte.", 9],
        ["La faute de quelqu'un d'autre, clairement.", 7],
        ["Un rythme de croisière agréable.", 5]
      ]}
    ];

    var MAXSUM = QUESTIONS.reduce(function (s, q) {
      return s + q.o.reduce(function (m, o) { return Math.max(m, o[1]); }, 0);
    }, 0);

    var DELIB = [
      "Vérification de l'adresse au registre…",
      'Croisement avec les six éditions précédentes…',
      "Analyse de l'aptitude logistique déclarée…",
      'Consultation de Kransvelt… (pas de réponse)',
      'Délibération du Comité…'
    ];

    var gate = document.getElementById('gate');
    var gateForm = document.getElementById('gate-form');
    var gateErr = document.getElementById('gate-error');
    var quiz = document.getElementById('quiz');
    var quizHead = document.getElementById('quiz-head');
    var quizForm = document.getElementById('quiz-form');
    var quizQs = document.getElementById('quiz-questions');
    var quizErr = document.getElementById('quiz-error');
    var result = document.getElementById('result');
    var delib = document.getElementById('deliberation');
    var verdict = document.getElementById('verdict');
    if (!gateForm || !quizForm || !CFG) { return; }

    var me = { name: null, nh: null, eh: null };

    function sha40(salt, value) {
      var data = new TextEncoder().encode(salt + '|' + String(value).trim().toLowerCase());
      return crypto.subtle.digest('SHA-256', data).then(function (buf) {
        var b = new Uint8Array(buf), out = '';
        for (var i = 0; i < b.length; i++) {
          out += b[i].toString(16).padStart(2, '0');
        }
        return out.slice(0, 40);
      });
    }

    function store(key, val) {
      try { localStorage.setItem(key, val); } catch (e) { /* navigation privée */ }
    }
    function recall(key) {
      try { return localStorage.getItem(key); } catch (e) { return null; }
    }

    function buildQuiz() {
      QUESTIONS.forEach(function (item, qi) {
        var fs = document.createElement('fieldset');
        fs.className = 'question';
        var lg = document.createElement('legend');
        lg.textContent = (qi + 1) + '. ' + item.q;
        fs.appendChild(lg);
        var wrap = document.createElement('div');
        wrap.className = 'opts';
        item.o.forEach(function (opt, oi) {
          var id = 'q' + qi + 'o' + oi;
          var lab = document.createElement('label');
          lab.className = 'opt';
          lab.setAttribute('for', id);
          var input = document.createElement('input');
          input.type = 'radio';
          input.name = 'q' + qi;
          input.id = id;
          input.value = String(opt[1]);
          var sp = document.createElement('span');
          sp.textContent = opt[0];
          lab.appendChild(input);
          lab.appendChild(sp);
          wrap.appendChild(lab);
        });
        fs.appendChild(wrap);
        quizQs.appendChild(fs);
      });
    }

    function label(score) {
      if (score >= 85) { return 'Lucidité rare. Le Comité a relu deux fois.'; }
      if (score >= 70) { return 'Profil solide, légèrement trop raisonnable.'; }
      if (score >= 55) { return "Perfectible. On a vu pire, et on s'en souvient."; }
      return "Le Comité s'inquiète, et il a l'habitude.";
    }

    function showVerdict(score, retenu) {
      var box = document.createElement('div');
      box.className = 'verdict-box' + (retenu ? ' retenu' : '');

      var sw = document.createElement('div');
      sw.className = 'score-wrap';
      var n = document.createElement('div');
      n.className = 'score-num';
      n.textContent = String(score);
      var m = document.createElement('div');
      m.className = 'score-max';
      m.textContent = "/ 100 — indice d'aptitude organisationnelle";
      sw.appendChild(n); sw.appendChild(m);

      var lb = document.createElement('p');
      lb.className = 'score-label';
      lb.textContent = label(score);

      var h = document.createElement('p');
      h.className = 'headline';
      var p1 = document.createElement('p');
      var p2 = document.createElement('p');

      if (retenu) {
        h.textContent = me.name + ' — exactement ce dont le CO a besoin.';
        p1.textContent = "Score enregistré, profil retenu. Tu fais partie du comité d'organisation de la première édition de la saison II. À partir de maintenant, tu n'es plus un participant : tu es le problème de tous les autres.";
        p2.textContent = "Tu recevras tes instructions et ton accès par un autre canal. Tu n'organises pas seul, et tu ne sais pas encore avec qui : c'est voulu. N'en parle à personne, pas même à la personne avec qui tu t'apprêtes à en parler.";
      } else {
        h.textContent = 'Dommage, ' + me.name + '. Tu dois encore faire des efforts.';
        p1.textContent = 'Score enregistré, profil non retenu pour cette édition. Le Comité te remercie de ta candidature et te souhaite un week-end dont tu ne sauras rien.';
        p2.textContent = "Tu peux contester. Ce sera classé. C'est tout l'intérêt d'un registre.";
      }

      verdict.innerHTML = '';
      box.appendChild(sw);
      box.appendChild(lb);
      box.appendChild(h);
      box.appendChild(p1);
      box.appendChild(p2);
      verdict.appendChild(box);
      verdict.hidden = false;
    }

    function runDeliberation(score, retenu) {
      delib.innerHTML = '';
      var k = 0;
      (function next() {
        if (k < DELIB.length) {
          var d = document.createElement('div');
          d.textContent = '› ' + DELIB[k];
          delib.appendChild(d);
          k++;
          setTimeout(next, 700);
        } else {
          showVerdict(score, retenu);
        }
      })();
    }

    function key() { return 'co-' + (CFG.edition || 'x') + '-' + me.nh; }

    gateForm.addEventListener('submit', function (ev) {
      ev.preventDefault();
      gateErr.hidden = true;
      var raw = document.getElementById('co-email').value;
      if (!raw || raw.indexOf('@') === -1) {
        gateErr.textContent = 'Il faut une adresse e-mail. Même Kransvelt en a une, paraît-il.';
        gateErr.hidden = false;
        return;
      }
      if (!window.crypto || !crypto.subtle) {
        gateErr.textContent = "Ce navigateur ne sait pas vérifier les empreintes (il faut une connexion https). Essaie depuis un autre appareil.";
        gateErr.hidden = false;
        return;
      }
      Promise.all([sha40(CFG.saltNames, raw), sha40(CFG.saltElect, raw)]).then(function (h) {
        var name = CFG.names[h[0]];
        if (!name) {
          gateErr.textContent = "Cette adresse ne figure pas au registre. Soit tu t'es trompé, soit tu es Kransvelt. Les deux sont graves.";
          gateErr.hidden = false;
          return;
        }
        me.name = name; me.nh = h[0]; me.eh = h[1];

        var saved = recall(key());
        if (saved) {
          var parts = saved.split('|');
          gate.hidden = true;
          result.hidden = false;
          delib.innerHTML = '';
          var d = document.createElement('div');
          d.textContent = "› Le Comité ne délibère qu'une fois par adresse. Verdict déjà rendu.";
          delib.appendChild(d);
          showVerdict(parseInt(parts[0], 10), parts[1] === '1');
          result.scrollIntoView({ behavior: 'smooth', block: 'start' });
          return;
        }

        gate.hidden = true;
        quizHead.textContent = "Étape 2 — Évaluation d'aptitude · candidat : " + name;
        quiz.hidden = false;
        quiz.scrollIntoView({ behavior: 'smooth', block: 'start' });
      })['catch'](function () {
        gateErr.textContent = 'La vérification a échoué. Réessaie.';
        gateErr.hidden = false;
      });
    });

    quizForm.addEventListener('submit', function (ev) {
      ev.preventDefault();
      quizErr.hidden = true;
      var sum = 0, answered = 0;
      for (var i = 0; i < QUESTIONS.length; i++) {
        var sel = quizForm.querySelector('input[name="q' + i + '"]:checked');
        if (sel) { answered++; sum += parseInt(sel.value, 10); }
      }
      if (answered < QUESTIONS.length) {
        quizErr.textContent = 'Il reste ' + (QUESTIONS.length - answered) + ' question(s) sans réponse. Le Comité exige un dossier complet.';
        quizErr.hidden = false;
        return;
      }
      var score = Math.round((sum / MAXSUM) * 100);
      var retenu = CFG.elect.indexOf(me.eh) !== -1;
      store(key(), score + '|' + (retenu ? '1' : '0'));
      quiz.hidden = true;
      result.hidden = false;
      result.scrollIntoView({ behavior: 'smooth', block: 'start' });
      runDeliberation(score, retenu);
    });

    buildQuiz();
  })();
})();
