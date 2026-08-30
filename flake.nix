{
  description = "audio-stack-rs — host-agnostic Rust audio playback backend";

  # Same shared flake as ../janis: rust-projects owns the one deterministic
  # toolchain pin, and its dev shell materializes the canonical
  # rust-toolchain.toml at the project root. Do not add one here.
  inputs.rust-projects.url = "github:obazin/rust-projects";

  outputs =
    { rust-projects, ... }:
    rust-projects.forEachSystem (
      lib:
      let
        inherit (lib) pkgs;

        # The two build requirements from the README, on top of the shared
        # Rust toolchain:
        #
        # cmake: the bundled Opus decoder builds libopus from source via
        # `opusic-sys`, which drives cmake. Without it the backend won't
        # compile. (The C toolchain it also needs comes from the stdenv.)
        #
        # alsa-lib + pkg-config: cpal's Linux backend links ALSA through
        # pkg-config. macOS reaches CoreAudio through the SDK and needs
        # nothing here.
        nativeDeps = [ pkgs.cmake ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.pkg-config ];
        buildDeps = pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.alsa-lib ];
      in
      lib.mkRustProject {
        name = "audio-stack-rs";
        src = ./.;

        # `extra` puts these on PATH in the dev shell; nativeBuildInputs /
        # buildInputs give the same deps to the package + checks builds.
        extra = nativeDeps ++ buildDeps;
        nativeBuildInputs = nativeDeps;
        buildInputs = buildDeps;
      }
    );
}
