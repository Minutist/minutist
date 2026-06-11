# Contributing

Thanks for considering a contribution. This project is small and the rules
below are short, however two of them (the DCO sign-off and the inbound
licence grant) are load-bearing for how the project is licensed, so please
read this whole file before opening a pull request.

These terms apply to the project under its production name (Minutist) and
its original working name (meeting-app); the rename does not change them.

## Licence

The repository is licensed under the GNU Affero General Public License,
version 3 only (AGPL-3.0-only); see [LICENSE](LICENSE). The licence map is
[REUSE.toml](REUSE.toml). The Minutist name and logo are trademarks and are
NOT covered by the code licence; see [TRADEMARKS.md](TRADEMARKS.md).

The project is dual-licensed in practice: the maintainer also distributes
builds of this code (and hosted services around it) under separate
commercial terms. The inbound grant below is what makes that possible while
keeping contributions simple, there is no CLA to sign.

## Developer Certificate of Origin (DCO)

Every commit must be signed off:

    git commit -s

The sign-off certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/):
that you wrote the contribution or
otherwise have the right to submit it under the project's licence. Use your
real name and a working email address. Commits without a `Signed-off-by:`
line will not be merged.

## Inbound licence grant

By submitting a contribution (a pull request, patch, or any other material
intended for inclusion in the project) you agree to the following terms.
Your `Signed-off-by:` line records both your DCO certification and your
agreement to this grant.

1. **Copyright.** You license the contribution under AGPL-3.0-only, and
   additionally you grant Andrew Leech, and any successor maintainer or
   entity to which the project is formally assigned, a perpetual,
   worldwide, non-exclusive, irrevocable, royalty-free licence to
   reproduce, modify, adapt, publish, distribute, publicly perform and
   display, and sublicense the contribution and derivative works of it,
   **under any licence terms, including proprietary and commercial
   terms**, as part of this project or works derived from it.
2. **Patents.** You grant the same parties a perpetual, worldwide,
   non-exclusive, irrevocable, royalty-free patent licence, under any
   patent claims you own or can license that are necessarily infringed by
   your contribution alone or by its combination with the project, to
   make, have made, use, offer to sell, sell, import, and otherwise
   transfer the contribution and such combinations.
3. **Moral rights.** To the maximum extent permitted by law, you consent
   to acts and omissions by the project and its licensees in relation to
   your contribution that would otherwise infringe your moral rights
   (including use without attribution and modification or adaptation of
   the contribution). This is a consent, not an assignment.
4. **Authority.** The contribution is your own original work, or you
   otherwise have the right to grant all of the above over all of it. If
   you made it in the course of employment, you confirm your employer has
   authorised the contribution under these terms or waived its rights in
   it. If it incorporates third-party material, identify that material in
   the pull request. Note that material submitted under DCO clause (b)
   (covered by another open-source licence) generally **cannot** be
   granted under proprietary terms and will be handled case by case or
   declined.

You keep the copyright in your contribution. This grant is not exclusive:
you can reuse your own work however you like. What the grant does is let
the project continue to ship commercial builds and relicense as a whole
without having to track down every past contributor, which (as the
Standard Notes relicensing episode showed) is otherwise nigh on impossible
once a project has history.

If you cannot or do not want to make this grant, that's fine, but the
contribution can't be merged.

## Practical notes

- Read `architecture/README.md` before changing code; component boundaries
  and dependency rules are enforced by a pre-commit hook and by review.
- Run `make test` (Rust workspace + UI) before pushing.
- No `println!` outside test code, no `anyhow::Error` in public signatures,
  no unbounded channels; the full set is in `architecture/cross-cutting.md`.
