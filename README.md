# Sonus

Identify and extract orchestral instruments

## Project Tree

```
Sonus
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ config
│  ├─ description
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           └─ main
│  ├─ objects
│  │  ├─ 01
│  │  │  └─ e232b714745a1906abf900793b526caa732f27
│  │  ├─ 05
│  │  │  ├─ 6aefa8e653e908cf4767577374e480362e0ea4
│  │  │  └─ 8761b3c14294ac3e1f5f2f04aa114fd28cf90d
│  │  ├─ 06
│  │  │  └─ a39ac0ed29fafb8e9d1f514c2b4523b9a93c72
│  │  ├─ 43
│  │  │  └─ c3fd562435c79c61958167eef5b82b8f2fcb98
│  │  ├─ 45
│  │  │  └─ ce68a33e63d464c5c86ea25868042d0b656201
│  │  ├─ 5e
│  │  │  └─ 00c2d7c1b1d8c3b628cf01ceacaccd314415b2
│  │  ├─ 5f
│  │  │  └─ 4fff14b5ac1b14b26400412d6d1a9624ed2e2f
│  │  ├─ 63
│  │  │  └─ 883be63d7ee9438e21c2687a0f7344ccf768ce
│  │  ├─ 82
│  │  │  └─ fb5a2e9710ca58788fe20e11f24239ac31a880
│  │  ├─ 83
│  │  │  └─ 70c8e3d99a34edcc99fd742210b8e31f10e165
│  │  ├─ 8b
│  │  │  ├─ cdcd4de6797166e3244fe506d4f4e68d94e692
│  │  │  └─ dc90e714f603dedf69cfee153c9b893ff1ea4f
│  │  ├─ 91
│  │  │  └─ 9e3f5f17c24f5580858c7e12df9393caa411ae
│  │  ├─ a1
│  │  │  └─ 032e132025cc81bdd92700001d77425b25d5b0
│  │  ├─ a5
│  │  │  └─ ba9e38bd21393f4c80c79e5a470a6e01916970
│  │  ├─ a8
│  │  │  └─ 53eb2a99cdf45d9b16088919b20aaf3dfdebb7
│  │  ├─ b4
│  │  │  └─ 0002e4c170b40e9d9c92a36af47b9ed3e6748a
│  │  ├─ b6
│  │  │  └─ ddafd66d3df141665e3433f1c43b58451d9be6
│  │  ├─ c2
│  │  │  └─ 0689122cc63d0363f1c890f288cc8949ff9316
│  │  ├─ e1
│  │  │  └─ d2c313bcb32f6ad51f0d175f7dffce73d2166d
│  │  ├─ e3
│  │  │  └─ e15bd2bbb7ac4634962b723f63567e1aa25448
│  │  ├─ e4
│  │  │  └─ 381b09730afe521318b34518ee68280fce312d
│  │  ├─ e9
│  │  │  └─ 4e750f98dcfb55ad05b2ea6b0e195d6cf588af
│  │  ├─ ef
│  │  │  └─ af4cda39238e5dd918bc018eb91ea972a8743e
│  │  ├─ info
│  │  └─ pack
│  └─ refs
│     ├─ heads
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     └─ main
│     └─ tags
├─ .gitignore
├─ README.md
├─ backup
│  ├─ README.md
│  ├─ mode_create_241010.py
│  ├─ model_create.py
│  └─ model_create_each.py
├─ dataset_create_mp3.py
├─ logs
│  ├─ bass_clarinet
│  │  ├─ train
│  │  │  └─ events.out.tfevents.1728222069.Reoui-MacBookPro.local.76344.0.v2
│  │  └─ validation
│  │     └─ events.out.tfevents.1728222261.Reoui-MacBookPro.local.76344.1.v2
│  ├─ bass_clarinet_training.log
│  └─ bassoon
├─ model_create.py
├─ model_create_each.py
├─ models
└─ 논문
   ├─ Deep Learning Methods for Instrument Separation and Recognition (Page 91~).pdf
   ├─ Deep Learning Methods for Instrument Separation and Recognition.pdf
   ├─ ch4. Deep Learning Methods for Instrument Separation and Recognition.pdf
   └─ 오케스트라 연주에서 딥러닝 기반 악기 식별.pdf

```
