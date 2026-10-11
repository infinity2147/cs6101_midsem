# IEOR DAMP website: Version 1

The IEOR Department Academic Mentorship Programme site for IIT Bombay. It is one place where an
IEOR student finds what their seniors know about courses, rules and opportunities, before they need it.

Built with Jekyll for GitHub Pages, the same stack as the Mech and EP DAMP sites. There is no server
and no build-time plugins. A small amount of plain JavaScript handles the filters and the calendar.

## What is in Version 1

| Feature | Where it lives | Status |
|---|---|---|
| **F1** Tagged post stream: one post collection; every list page is a filter over it, with filters for type, domain, writer's year and writer's programme, plus a tag cloud | `_posts/`, `_includes/post-stream.html`, `_includes/post-card.html`, `assets/js/site.js`, `/posts/` | Done |
| **F2** Structured course reviews: fixed header, an EP-style explainer of categories, courses grouped by semester, a per-course page with all reviews newest first, a side-by-side table and a "Running this semester" badge | `_courses/`, `_layouts/course.html`, `_includes/review-header.html`, `pages/courses.html` | Done; content needed |
| **F3** Course Tracks: 4 drafts (OR/optimization, supply chain, analytics/DS, quant finance); related posts and Radar entries filled in automatically by domain | `pages/tracks/`, `_layouts/track.html` | Drafted; each needs a senior's check |
| **F4** Opportunity Radar: list, calendar and year-roadmap views; the home page "Open now" strip; all required fields; experience posts link back to their entry | `_posts/radar/`, `pages/radar.html`, `assets/js/radar.js` | 20 entries; months unverified |
| **F5** Rules and FAQs: 25 EP-style FAQs in 7 topics, each with an official link and a last-verified line | `_data/faqs.yml`, `pages/faq.html` | Written; each needs verifying |
| **F6** Team and Ask us: team cards, an Ask us form with an anonymous option, past teams | `_data/team.yml`, `pages/team.html`, `_includes/ask-form.html` | Needs real names and the Google Form |

Also from the spec: the home page uses EP's Guide / Common Room split, with section cards (one line
and one button), "Open now", the latest 5 posts, quick links and the Ask us form. The site has light
and dark themes, is mobile first, and the content model follows spec section 4.2.

**Deliberately not built (Version 2 and 3):** prof pages and AMAs, the seniors network, skill roadmaps,
the Recovery corner, "What didn't work" posts, gated resources, freshness badges, search, the
contribution pipeline, the planner, the events archive and the reading list. The tag list already
reserves their post types (`ug-project`, `further-studies`, `event`, `didnt-work`), so adding them later
needs no change to the content model.

## Run it locally

```bash
gem install bundler        # once
bundle install             # once
bundle exec jekyll serve   # http://localhost:4000
ruby scripts/check_posts.rb   # validate posts, FAQs, courses; prints v1 progress
```

## Add a post

1. Copy a template from `templates/` (course review, experience, radar entry).
2. Save it as `_posts/<folder>/YYYY-MM-DD-short-name.md`:
   - `reviews/` for course reviews
   - `experiences/` for internships, research, in-sem programmes and exchanges
   - `radar/` for Radar entries
   The folder only keeps things tidy. The `type` field decides which section a post appears in.
3. Fill every field and write freely below the front matter. Use only values from `_data/tags.yml`.
4. Run `ruby scripts/check_posts.rb`. It fails on a missing required field, such as the `channel` on
   an experience, or on a tag that is not in `tags.yml`.
5. Open a pull request. One person other than the writer reviews it before merging.

Rules that the layouts enforce:

- **Experience posts** open with the writer's year and when they did it, and always show "Found it through" (`channel`).
- **Course reviews** show the full header (workload, grading, difficulty 1–5, take it if / skip it if, last verified). The course page finds a review by the code at the start of `course:`.
- **Radar entries** need `opens` and `closes` as full month names. Experience posts with `radar: <entry-slug>` are listed on the entry's page automatically.
- **`last_verified`** is shown everywhere. Leave it empty until someone has checked the content; the page then says "Not yet verified".
- **Anonymous posts**: set `anonymous: true` and leave out `author`.

**Tags:** one team member owns `_data/tags.yml`. Add a value there before using it, so we never end up
with `supply-chain`, `scm` and `supplychain` as three tags.

**Courses:** one file per course in `_courses/`. Set `running: [Autumn 2026]` on the courses offered this
semester, and update `current_semester` in `_config.yml` each semester.

## Before launch

- [ ] Delete the three `sample: true` posts in `_posts/reviews/` and `_posts/experiences/` (`check_posts.rb` lists them).
- [ ] Add the IEOR B.Tech **core** courses to `_courses/` from the official curriculum. Only IE 101 is listed, and that comes from a secondary source. The electives' codes and names come from Mech DAMP's IEOR track; confirm them and set `last_verified`.
- [ ] Verify the opening and closing months and the eligibility of all 20 Radar entries, especially IEOR eligibility for the bank, consulting and research programmes; set `last_verified`. The months are the team's estimates.
- [ ] Verify each FAQ against its official source and set `last_verified`. Confirm the honours and minor rules for the 2024+ batches, and whether a DD exists (spec open question 1).
- [ ] Have a senior who followed a similar path check each Course Track. Then set `status:` to something other than `draft` and fill `checked_by` and `last_verified`.
- [ ] Check the links in `_data/links.yml`. They could not be opened from the build environment, and the academic calendar and timetable URLs in particular should be checked.
- [ ] Fill `_data/team.yml` (names, photos in `assets/img/team/`, faculty coordinator) and set `team_email` in `_config.yml`.
- [ ] Create the Ask us Google Form, owned by the team email, with four questions: name, contact, year, question. Put its `formResponse` URL and entry ids under `ask_form` in `_config.yml`. Until then, the form opens an email instead.
- [ ] Write the first real posts: course reviews for every core course done so far, 15 experience posts, and 3–5 SLP posts.

## Deploy

1. Create the repository under a **DAMP-owned GitHub organisation**, not a personal account, and copy the contents of this folder to its root.
2. Go to Settings → Pages → Source and choose **GitHub Actions**.
3. `.github/workflows/pages.yml` runs `check_posts.rb`, builds with the right base URL and deploys on every push to `main`. Pull requests are built and checked but not deployed.

For a custom domain (spec open question 5), add a `CNAME` file and set `url` in `_config.yml`.

## Handover (every year)

The outgoing coordinators:

1. Transfer admin of the GitHub organisation and the Google Forms to the new team.
2. Move the current team in `_data/team.yml` under `past:` and add the new team. The footer reads the year from there.
3. Run a freshness pass: re-verify, or update `last_verified` on, every post, FAQ and Radar entry older than two years. Update `current_semester` and the `running` badges.
4. Walk the new team through this README.

Put these steps in the DAMP coordinator handover checklist as well, not only in this repo.

## Layout

```
_posts/          all posts: reviews/, experiences/, radar/
_courses/        one file per course (the course pages)
_data/           tags.yml (official tag list), team.yml, faqs.yml, links.yml, nav.yml, categories.yml
_layouts/        default, page, post (switches header by type), course, track
_includes/       cards, headers per post type, filter stream, ask form, radar data
pages/           fixed pages: courses, tracks/, radar, experiences, all posts, FAQs, team
templates/       copy-paste post templates (not published)
scripts/         check_posts.rb
assets/          css/main.css, js/site.js, js/radar.js, img/
```

## What we borrowed, from the source repositories

The live DAMP sites could not be opened from the build environment, so these notes come from the
sites' source repositories:

- **Mech DAMP** (`mechdampiitb.github.io`, Jekyll): a tag-driven nav with Courses, Course Tracks,
  Experiences, Project Experiences, In-Sem Opportunities, Further Studies, Events and Team. In-sem
  experience posts start with "How did you first come to know about the program?". Its IEOR minor track
  gave the elective codes and names seeded in `_courses/`. The footer still says "Mech DAMP Team 2023-24"
  (gap G8), which is why `team.yml` drives our footer.
- **EP DAMP** (`epdampiitb.github.io`, Jekyll): a home page with "The Hitchhiker's Guide" and "The
  Common Room" tile groups (one line plus a call to action per tile), Quick Links (department, academic
  homepage, calendar and timetable, ASC), FAQ sections for General, Minor and Honours, course reviews
  with code, category, workload and grading in the front matter, and team cards from a data file.
- **DESE DAMP** is a WordPress site and could not be loaded. Its coverage was taken from the spec.
  **Chemistry Fam** could not be found; its one-stream, category-plus-tags model is what F1 implements.
