#!/usr/bin/env ruby
# Checks every post, FAQ and course against the content model (spec sections 4.2 and 8),
# then prints progress towards the Version 1 "done when" targets.
#
#   ruby scripts/check_posts.rb          # from the site folder
#
# Exits with status 1 if any post breaks a rule, so CI can block the merge.
require "yaml"
require "date"

ROOT = File.expand_path("..", __dir__)
MONTHS = Date::MONTHNAMES.compact
EXPERIENCE_TYPES = %w[internship research in-sem semex].freeze

def load_yaml(path)
  YAML.safe_load(File.read(path, encoding: "UTF-8"), permitted_classes: [Date, Time], aliases: true)
end

def front_matter(path)
  text = File.read(path, encoding: "UTF-8")
  m = text.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  return nil unless m
  YAML.safe_load(m[1], permitted_classes: [Date, Time]) || {}
end

tags = load_yaml(File.join(ROOT, "_data/tags.yml"))
links = load_yaml(File.join(ROOT, "_data/links.yml"))
faqs = load_yaml(File.join(ROOT, "_data/faqs.yml"))
errors = []
warnings = []

blank = ->(v) { v.nil? || (v.respond_to?(:empty?) && v.empty?) }
need = lambda do |file, fm, keys|
  keys.each { |k| errors << "#{file}: missing `#{k}`" if blank.(fm[k]) }
end
allowed = lambda do |file, fm, key, list_name = key|
  v = fm[key]
  return if blank.(v)
  ok = tags[list_name].is_a?(Hash) ? tags[list_name].key?(v.to_s) : Array(tags[list_name]).include?(v.to_s)
  errors << "#{file}: `#{key}: #{v}` is not in _data/tags.yml -> #{list_name}" unless ok
end

posts = Dir[File.join(ROOT, "_posts/**/*.md")].sort.map do |path|
  rel = path.sub("#{ROOT}/", "")
  fm = front_matter(path)
  if fm.nil?
    errors << "#{rel}: no front matter"
    next
  end
  unless File.basename(path) =~ /\A\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md\z/
    errors << "#{rel}: file name must be YYYY-MM-DD-lowercase-words.md"
  end
  fm["type"] ||= "radar" if rel.start_with?("_posts/radar/")
  [rel, fm]
end.compact

radar_slugs = posts.select { |_, fm| fm["type"] == "radar" }
                   .map { |rel, _| File.basename(rel, ".md").sub(/\A\d{4}-\d{2}-\d{2}-/, "") }

posts.each do |rel, fm|
  type = fm["type"]
  need.(rel, fm, %w[type title])
  allowed.(rel, fm, "type")
  allowed.(rel, fm, "domain")
  allowed.(rel, fm, "author_year")
  allowed.(rel, fm, "author_program")
  Array(fm["tags"]).each do |t|
    errors << "#{rel}: tag `#{t}` is not in _data/tags.yml -> tags" unless tags["tags"].include?(t.to_s)
  end
  errors << "#{rel}: `last_verified` key is required (leave empty if not checked yet)" unless fm.key?("last_verified")
  unless type == "radar" || fm["anonymous"] || !blank.(fm["author"])
    errors << "#{rel}: needs `author`, or `anonymous: true`"
  end

  case type
  when "course-review"
    need.(rel, fm, %w[course instructor semester category prerequisites workload_hours_per_week grading difficulty take_it_if skip_it_if author_year])
    allowed.(rel, fm, "category")
    unless fm["course"].to_s =~ /\A[A-Z]{2,3} ?\d{3}[A-Z]? - .+/
      errors << "#{rel}: `course` must look like `IE 630 - Course name`"
    end
    d = fm["difficulty"]
    errors << "#{rel}: `difficulty` must be 1 to 5" if d && !(1..5).cover?(d.to_i)
  when *EXPERIENCE_TYPES
    need.(rel, fm, %w[org role domain author_year author_program when_done channel])
    allowed.(rel, fm, "channel")
    if fm["radar"] && !radar_slugs.include?(fm["radar"].to_s)
      errors << "#{rel}: `radar: #{fm['radar']}` does not match any _posts/radar entry"
    end
  when "radar"
    need.(rel, fm, %w[name organiser kind eligibility channel opens closes stages leads_to roadmap])
    allowed.(rel, fm, "kind")
    allowed.(rel, fm, "channel")
    %w[opens closes].each do |k|
      errors << "#{rel}: `#{k}: #{fm[k]}` must be a full month name" if fm[k] && !MONTHS.include?(fm[k].to_s)
    end
    Array(fm["roadmap"]).each do |s|
      errors << "#{rel}: roadmap stage `#{s}` is not in _data/tags.yml -> roadmap" unless tags["roadmap"].key?(s.to_s)
    end
  end
  warnings << "#{rel}: sample post, delete before launch" if fm["sample"]
end

faq_count = 0
faqs.each do |topic|
  Array(topic["items"]).each do |item|
    faq_count += 1
    where = "_data/faqs.yml (#{topic['id']}): #{item['q']}"
    errors << "#{where}: missing answer" if blank.(item["a"])
    errors << "#{where}: `source: #{item['source']}` is not in _data/links.yml -> official" unless links["official"].key?(item["source"].to_s)
  end
end

courses = Dir[File.join(ROOT, "_courses/*.md")].map { |p| [p.sub("#{ROOT}/", ""), front_matter(p) || {}] }
courses.each do |rel, fm|
  need.(rel, fm, %w[code name category group])
  allowed.(rel, fm, "category")
end

tracks = Dir[File.join(ROOT, "pages/tracks/*.md")].map { |p| front_matter(p) || {} }

# ---- report ----
real = posts.reject { |_, fm| fm["sample"] }
radar = posts.select { |_, fm| fm["type"] == "radar" }
reviews = real.select { |_, fm| fm["type"] == "course-review" }
reviewed_codes = reviews.map { |_, fm| fm["course"].to_s.split(" - ").first.to_s.strip }
core = courses.select { |_, fm| fm["category"] == "core" }
core_done = core.count { |_, fm| reviewed_codes.include?(fm["code"].to_s) }
radar_verified = radar.count { |_, fm| !blank.(fm["last_verified"]) }

puts "Version 1 progress (spec section 5, 'Done when')"
puts "  F2 core courses with a real review : #{core_done}/#{core.size} (core list itself still to be completed)"
puts "  F3 tracks checked by a senior      : #{tracks.count { |t| t['status'] != 'draft' }}/4 (#{tracks.size} drafted)"
puts "  F4 radar entries with months       : #{radar.count { |_, fm| fm['opens'] && fm['closes'] }}/20 (#{radar_verified} verified)"
puts "  F5 FAQs with an official link      : #{faq_count}/25"
puts "  Posts (excluding samples)          : #{real.size - radar.size}"
puts
warnings.each { |w| puts "warning: #{w}" }
if errors.empty?
  puts "OK: #{posts.size} posts, #{courses.size} courses, #{faq_count} FAQs checked."
else
  errors.each { |e| puts "error: #{e}" }
  puts "#{errors.size} error(s)."
  exit 1
end
