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

    glslang = {
      url = "github:KhronosGroup/glslang/2d8b71fc63578a93726c05e0565c3ef064bdc1ba";
      flake = false;
    };
    spirv-tools = {
      url = "github:KhronosGroup/SPIRV-Tools/0cfe9e7219148716dfd30b37f4d21753f098707a";
      flake = false;
    };
    spirv-headers = {
      url = "github:KhronosGroup/SPIRV-Headers/2acb319af38d43be3ea76bfabf3998e5281d8d12";
      flake = false;
    };
    shaderc = {
      url = "github:google/shaderc/v2024.2";
      flake = false;
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

      shaderc = stdenv.mkDerivation rec {
        name = "shaderc";
        src = inputs.shaderc;

        dontStrip = true;

        outputs = [ "out" "lib" "bin" "dev" "static" ];

        postPatch = ''
          cp -r --no-preserve=mode ${inputs.glslang} third_party/glslang
          cp -r --no-preserve=mode ${inputs.spirv-tools} third_party/spirv-tools
          ln -s ${inputs.spirv-headers} third_party/spirv-tools/external/spirv-headers
          patchShebangs --build utils/
        '';

        nativeBuildInputs = [ pkgs.cmake pkgs.python3 ];

        postInstall = ''
          moveToOutput "lib/*.a" $static
        '';

        cmakeFlags = [ "-DSHADERC_SKIP_TESTS=ON" ];

        # Fix the paths in .pc, even though it's unclear if all these .pc are really useful.
        postFixup = ''
          substituteInPlace "$dev"/lib/pkgconfig/*.pc \
            --replace '=''${prefix}//' '=/' \
            --replace "$dev/$dev/" "$dev/"
        '';

        meta = with pkgs.lib; {
          inherit (src.meta) homepage;
          description = "A collection of tools, libraries and tests for shader compilation";
          platforms = platforms.all;
          license = [ licenses.asl20 ];
        };
      };
      glslang = stdenv.mkDerivation rec {
        name = "glslang";

        src = inputs.glslang;

        dontStrip = true;

        # These get set at all-packages, keep onto them for child drvs
        # passthru = {
        #   spirv-tools = pkgs.spirv-tools;
        #   spirv-headers = pkgs.spirv-headers;
        # };

        nativeBuildInputs = with pkgs; [cmake python3 bison jq];

        postPatch = ''
          cp --no-preserve=mode -r "${inputs.spirv-tools}" External/spirv-tools
          ln -s "${inputs.spirv-headers}" External/spirv-tools/external/spirv-headers
        '';

        # This is a dirty fix for lib/cmake/SPIRVTargets.cmake:51 which includes this directory
        postInstall = ''
          mkdir $out/include/External
        '';

        # Fix the paths in .pc, even though it's unclear if these .pc are really useful.
        postFixup = ''
          substituteInPlace $out/lib/pkgconfig/*.pc \
            --replace '=''${prefix}//' '=/'

          # add a symlink for backwards compatibility
          ln -s $out/bin/glslang $out/bin/glslangValidator
        '';

        meta = with pkgs.lib; {
          inherit (src.meta) homepage;
          description = "Khronos reference front-end for GLSL and ESSL";
          license = licenses.asl20;
          platforms = platforms.unix;
          maintainers = [maintainers.ralith];
        };
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
          curl
          fswatch

          # - [nixOS usage | Mach: zig game engine & graphics toolkit](https://machengine.org/about/nixos-usage/)
          xorg.libX11
          vulkan-loader
        ])
        ++ [
          pkgs.glslang
          pkgs.shaderc
          # glslang
          # shaderc
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
