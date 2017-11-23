# Configuring the archiver host

The software on both the instrument and archiver computers run under Centos LINUX.

The user is **hsrl**.
Set the hsrl user login shell to **tcsh**.
Make sure **python** is installed.

To install the LROSE software, see:

  [LROSE core](https://ncar.github.io/lrose-core)

To configure the runtime environment on the HSRL archiver host, first check out hsrl_configuration:

```bash
  mkdir -p ~/git
  cd ~/git
  git clone https://github.com/ncar/hsrl_configuration
```

Make the data directory, and give it the correct permissions:

```bash
  sudo mkdir -p /data/hsrl
  sudo chown -R hsrl /data/hsrl
```

Go to the systems/scripts directory, and run the configuration script:

```bash
  cd ~/git/hsrl_configuration/projDir/system/scripts
  ./configureHost.py
```

Source the .cshrc file

```csh
  source ~/.cshrc
```

Go to the projects directory:

```csh
  cd ~/projDir
```
