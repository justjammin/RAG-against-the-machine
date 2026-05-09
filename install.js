#!/usr/bin/env node

/**
 * Default: copy each SKILL.md → ~/.claude/skills/<id>/SKILL.md (Claude Code; Lena-style).
 * --project / --user: symlink full skill folders into .claude/skills and/or .cursor/skills.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const ROOT = __dirname;

const SKILLS = [{ id: 'ratm', skillDir: path.join(ROOT, 'skill', 'ratm'), src: path.join(ROOT, 'skill', 'ratm', 'SKILL.md') }];

function printHelp() {
  console.log(`
Usage: node install.js [options]

  (no flags)     Copy SKILL.md files to ~/.claude/skills/<id>/ (requires ~/.claude)

  --project      Symlink full skill dirs into ./.claude/skills and ./.cursor/skills
                 (run from the target project root — uses cwd)

  --user         Symlink into ~/.claude/skills and ~/.cursor/skills

  --claude-only  With --project or --user: only .claude/skills
  --cursor-only  With --project or --user: only .cursor/skills

  -h, --help     This message
`);
}

function copyToUserClaude() {
  if (!fs.existsSync(path.join(os.homedir(), '.claude'))) {
    console.error(
      'Error: Claude Code not found. Install from https://claude.ai/code first.'
    );
    process.exit(1);
  }

  for (const { id, src } of SKILLS) {
    if (!fs.existsSync(src)) {
      console.error(`Error: missing ${src}`);
      process.exit(1);
    }
    const dir = path.join(os.homedir(), '.claude', 'skills', id);
    const dest = path.join(dir, 'SKILL.md');
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.copyFileSync(src, dest);
    console.log(`  ${id} → ~/.claude/skills/${id}/SKILL.md`);
  }

  console.log('');
  console.log('ratm skill installed (copy).');

  // Attempt pip install for Python package + Claude integration
  try {
    const { execSync } = require('child_process');
    try {
      execSync('which python3', { stdio: 'pipe' });
    } catch (_) {
      console.warn('  Warning: python3 not found — skipping pip install. Run manually: pip install -e . && ratm install');
      printTail();
      return;
    }
    console.log('  Installing Python package (pip install -e . && ratm install)...');
    execSync('pip install -e . && ratm install', { cwd: ROOT, stdio: 'inherit' });
  } catch (err) {
    console.warn('  Warning: pip install failed — Python package not installed.');
    console.warn('  Run manually: pip install -e . && ratm install');
  }

  printTail();
}

function replaceSymlink(linkPath, targetPath) {
  if (fs.existsSync(linkPath)) {
    const st = fs.lstatSync(linkPath);
    if (st.isSymbolicLink()) {
      fs.unlinkSync(linkPath);
    } else {
      console.error(`Error: exists and is not a symlink (refusing to replace): ${linkPath}`);
      process.exit(1);
    }
  }
  fs.symlinkSync(targetPath, linkPath, 'dir');
}

function linkInto(skillsRoot, label) {
  if (!fs.existsSync(skillsRoot)) {
    fs.mkdirSync(skillsRoot, { recursive: true });
  }
  for (const { id, skillDir, src } of SKILLS) {
    if (!fs.existsSync(src)) {
      console.error(`Error: missing ${src}`);
      process.exit(1);
    }
    const linkPath = path.join(skillsRoot, id);
    replaceSymlink(linkPath, skillDir);
    console.log(`  ${linkPath} → ${skillDir}`);
  }
  console.log(`\nSymlinked into ${label}`);
}

function printTail() {
  console.log('');
  console.log('  Run: npx ratm-skills');
  console.log('  OR:  pip install -e . && ratm install');
  console.log('');
  console.log('  Invoke: match skill descriptions in Claude Code, or @-attach a SKILL.md');
  console.log('');
}

function main() {
  const args = process.argv.slice(2);
  if (args.includes('-h') || args.includes('--help')) {
    printHelp();
    process.exit(0);
  }

  const project = args.includes('--project');
  const user = args.includes('--user');
  const claudeOnly = args.includes('--claude-only');
  const cursorOnly = args.includes('--cursor-only');

  if (claudeOnly && cursorOnly) {
    console.error('Error: use at most one of --claude-only, --cursor-only');
    process.exit(1);
  }

  const doClaude = !cursorOnly;
  const doCursor = !claudeOnly;

  if (project && user) {
    console.error('Error: use either --project or --user, not both');
    process.exit(1);
  }

  if (project) {
    const base = process.cwd();
    if (doClaude) {
      linkInto(path.join(base, '.claude', 'skills'), path.join(base, '.claude/skills'));
    }
    if (doCursor) {
      linkInto(path.join(base, '.cursor', 'skills'), path.join(base, '.cursor/skills'));
    }
    printTail();
    return;
  }

  if (user) {
    const home = os.homedir();
    if (doClaude) {
      linkInto(path.join(home, '.claude', 'skills'), '~/.claude/skills');
    }
    if (doCursor) {
      linkInto(path.join(home, '.cursor', 'skills'), '~/.cursor/skills');
    }
    printTail();
    return;
  }

  copyToUserClaude();
}

main();
