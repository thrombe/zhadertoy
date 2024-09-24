{
  description = "yaaaaaaaaaaaaaaaaaaaaa";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    zls = {
      # - [mach needs zls fork :/](https://discord.com/channels/996677443681267802/996677444360736831/1283458075675590838)
      url = "github:slimsag/zls/2024.5.0-mach";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.zig-overlay.follows = "zig-overlay";
    };

    zig2nix = {
      url = "github:Cloudef/zig2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    mach-flake = {
      url = "github:Cloudef/mach-flake";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.zig2nix.follows = "zig2nix";
    };
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system: let
      flakePackage = flake: package: flake.packages."${system}"."${package}";
      flakeDefaultPackage = flake: flakePackage flake "default";

      # - [Zig version | Mach](https://machengine.org/docs/zig-version/)
      # - [Nominated Zig versions](https://machengine.org/docs/nominated-zig/)
      # - [zig-overlay/sources.json](https://github.com/mitchellh/zig-overlay/blob/main/sources.json)
      # - [mach-flake templates flake.nix](https://github.com/Cloudef/mach-flake/blob/eca800e7fa289e95d3c0f32284763ad6984bb0a9/templates/engine/flake.nix)
      mach-env = inputs.mach-flake.outputs.mach-env.${system} {};
      overlays = [
        (self: super: {
          zig = mach-env.zig;
          zls = (flakePackage inputs.zls "zls").overrideAttrs (old: {
            nativeBuildInputs = [
              mach-env.zig
            ];
            buildInputs = [
              mach-env.zig
            ];
          });
        })
      ];

      pkgs = import inputs.nixpkgs {
        inherit system;
        inherit overlays;
      };

      fhs = pkgs.buildFHSEnv {
        name = "fhs-shell";
        targetPkgs = p: (env-packages p) ++ (custom-commands p);
        runScript = "${pkgs.zsh}/bin/zsh";
        profile = ''
          export FHS=1
          # source ./.venv/bin/activate
          # source .env
        '';
      };
      custom-commands = pkgs: [
        (pkgs.writeShellScriptBin "todo" ''
          #!/usr/bin/env bash
          cd $PROJECT_ROOT
        '')
      ];

      env-packages = pkgs:
        (with pkgs; [
          pkg-config
          # - [river.nix nixpkgs](https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/applications/window-managers/river/default.nix#L41)
          zig
          zls
          gdb

          # - [nixOS usage | Mach: zig game engine & graphics toolkit](https://machengine.org/about/nixos-usage/)
          xorg.libX11
          vulkan-loader
        ])
        ++ [
          pkgs.glslang
          pkgs.shaderc
        ]
        ++ (custom-commands pkgs);

      stdenv = pkgs.clangStdenv;
      # stdenv = pkgs.gccStdenv;
    in {
      packages = {};
      overlays = {};

      devShells.default =
        pkgs.mkShell.override {
          inherit stdenv;
        } {
          nativeBuildInputs = (env-packages pkgs) ++ [fhs];
          inputsFrom = [];
          shellHook = ''
            export PROJECT_ROOT="$(pwd)"
            export LD_LIBRARY_PATH=${pkgs.xorg.libX11}/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
          '';
        };
    });
}
