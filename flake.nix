{
  description = "yaaaaaaaaaaaaaaaaaaaaa";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    zig2nix = {
      url = "github:Cloudef/zig2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };

    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    zls = {
      url = "github:slimsag/zls/2024.5.0-mach";
      # url = "github:zigtools/zls/0.13.0";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.zig-overlay.follows = "zig-overlay";
    };

    zig = {
      url = "github:ziglang/zig/64ef45eb059328ac8579c22506fa7574e8cf45f2";
      flake = false;
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

      zig-env = inputs.zig2nix.zig-env.${system};
      mach-zig = pkgs.callPackage "${inputs.zig2nix}/zig.nix" rec {
        version = "0.13.0-dev.351+64ef45eb0";
        release = version;
        zigSystem = (zig-env {}).lib.zigDoubleFromString system;
        zigHook = (zig-env {}).zig-hook;
      };
      env = inputs.mach-flake.outputs.mach-env.${system} {};

      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [
          (self: super: {
            # zig = super.zig.overrideAttrs (drv: {
            #   src = inputs.zig;
            #   version = "0.13.0-dev.351+64ef45eb0";
            #   pversion = "0.13.0-dev.351+64ef45eb0";
            # });
            # zig = mach-zig;
            zig = env.zig;
            zls = (flakePackage inputs.zls "zls").overrideAttrs (old: {
              nativeBuildInputs = [
                env.zig
                # (super.zig.overrideAttrs (drv: {
                #   src = inputs.zig;
                # }))
              ];
              buildInputs = [
                env.zig
              ];
            });
          })
        ];
      };

      # - [Zig version | Mach](https://machengine.org/docs/zig-version/)
      # - [Nominated Zig versions](https://machengine.org/docs/nominated-zig/)
      # - [zig-overlay/sources.json](https://github.com/mitchellh/zig-overlay/blob/main/sources.json)
      # zig = flakePackage inputs.zig-overlay "master-2024-06-01";

      # zls = (flakePackage inputs.zls "zls").overrideAttrs (old: {
      #   nativeBuildInputs = [zig];
      # });

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

          xorg.libX11
          vulkan-loader
        ])
        ++ [
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
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            export CLANGD_FLAGS="--compile-commands-dir=$PROJECT_ROOT/plugin --query-driver=$(which $CXX)"
          '';
        };
    });
}
