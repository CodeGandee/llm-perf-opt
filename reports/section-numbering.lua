-- Simple heading numbering filter for Pandoc.
-- Adds "1 ", "1.1 ", "1.1.1 ", ... prefixes to headings H2–H6.

local counters = {}

local function reset_from(level)
  for i = level, 6 do
    counters[i] = counters[i] or 0
  end
end

function Header(el)
  local level = el.level

  -- Do not number H1 titles; only number H2+.
  if level <= 1 or level > 6 then
    return el
  end

  reset_from(level)
  counters[level] = (counters[level] or 0) + 1

  -- Reset deeper levels.
  for i = level + 1, 6 do
    counters[i] = 0
  end

  -- Build prefix from levels 2..level (skip level 1).
  local parts = {}
  for i = 2, level do
    if counters[i] and counters[i] > 0 then
      table.insert(parts, tostring(counters[i]))
    end
  end

  local prefix = table.concat(parts, ".") .. " "
  table.insert(el.content, 1, pandoc.Str(prefix))
  return el
end
