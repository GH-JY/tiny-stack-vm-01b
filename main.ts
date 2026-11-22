// VM for make.fun's tiny stack ISA (version 1): runs each instance's program.bin on its
// input.bin and writes the bytes the program produced and the exit code it ended with.
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Opcode byte -> [immediate bytes, values popped, values pushed]. No entry: not an opcode.
const OPS: Record<number, number[]> = {
  0x00: [0, 0, 0], 0x01: [1, 0, 1], 0x02: [4, 0, 1], 0x03: [0, 1, 0],
  0x04: [0, 1, 2], 0x05: [0, 2, 2], 0x06: [0, 2, 3], 0x07: [0, 3, 3],
  0x10: [0, 2, 1], 0x11: [0, 2, 1], 0x12: [0, 2, 1], 0x13: [0, 2, 1],
  0x14: [0, 2, 1], 0x15: [0, 2, 1], 0x16: [0, 2, 1], 0x17: [0, 1, 1],
  0x18: [0, 2, 1], 0x19: [0, 2, 1], 0x1a: [0, 2, 1], 0x1b: [0, 1, 1],
  0x1c: [0, 2, 1], 0x1d: [0, 2, 1], 0x1e: [0, 2, 1], 0x20: [0, 2, 1],
  0x21: [0, 2, 1], 0x22: [0, 2, 1], 0x30: [2, 0, 0], 0x31: [2, 1, 0],
  0x32: [2, 1, 0], 0x33: [2, 0, 0], 0x34: [0, 0, 0], 0x40: [0, 1, 1],
  0x41: [0, 2, 0], 0x50: [0, 1, 0], 0x51: [0, 1, 0], 0x52: [0, 0, 1],
  0x53: [0, 1, 0],
};

// Runs one program to its end: the bytes it wrote and the code it ended with.
// Limits: 256 stack values, 64 return addresses, 256 memory words, 1e6 steps, 65536 bytes out.
function vm(p: Uint8Array, inp: Uint8Array): [Uint8Array, number] {
  const n = p.length;
  const s = new Int32Array(256);
  const k = new Int32Array(64);
  const m = new Int32Array(256);
  const o = new Uint8Array(65536);
  let sp = 0, cp = 0, ol = 0, ip = 0, pc = 0, t = 0;
  // A trap keeps what was written before it, and its code is the exit code.
  const stop = (code: number): [Uint8Array, number] => [o.subarray(0, ol), code];

  for (;;) {
    if (t >= 1e6) return stop(10);
    if (pc < 0 || pc >= n) return stop(9);
    const op = p[pc], spec = OPS[op];
    if (!spec) return stop(5);
    const [imm, pops, pushes] = spec;
    if (pc + imm >= n) return stop(9);
    t++;
    if (sp < pops) return stop(2);
    if (sp - pops + pushes > 256) return stop(3);
    const next = pc + 1 + imm;
    pc = next; // unless it jumps, pc already points at the next instruction

    switch (op) {
      case 0x00: return stop(0);
      case 0x01: s[sp++] = (p[next - 1] << 24) >> 24; break;
      case 0x02: {
        const i = next - 4;
        s[sp++] = p[i] | (p[i + 1] << 8) | (p[i + 2] << 16) | (p[i + 3] << 24);
        break;
      }
      case 0x03: sp--; break;
      case 0x04: s[sp] = s[sp - 1]; sp++; break;
      case 0x05: { const x = s[sp - 1]; s[sp - 1] = s[sp - 2]; s[sp - 2] = x; break; }
      case 0x06: s[sp] = s[sp - 2]; sp++; break;
      case 0x07: {
        const x = s[sp - 3];
        s[sp - 3] = s[sp - 2]; s[sp - 2] = s[sp - 1]; s[sp - 1] = x;
        break;
      }
      case 0x17: s[sp - 1] = -s[sp - 1] | 0; break;
      case 0x1b: s[sp - 1] = ~s[sp - 1]; break;
      // The binary operators: pop b then a, push one result.
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16:
      case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d: case 0x1e:
      case 0x20: case 0x21: case 0x22: {
        const b = s[sp - 1], a = s[sp - 2];
        let r = 0;
        switch (op) {
          case 0x10: r = (a + b) | 0; break;
          case 0x11: r = (a - b) | 0; break;
          case 0x12: r = Math.imul(a, b); break;
          case 0x13: // the overflowing quotient stays at the most negative value
            if (b === 0) return stop(4);
            r = a === -2147483648 && b === -1 ? a : (a / b) | 0;
            break;
          case 0x14: // the remainder takes the sign of a, and x % -1 is 0
            if (b === 0) return stop(4);
            r = b === -1 ? 0 : (a % b) | 0;
            break;
          case 0x15:
            if (b === 0) return stop(4);
            r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
            break;
          case 0x16:
            if (b === 0) return stop(4);
            r = ((a >>> 0) % (b >>> 0)) | 0;
            break;
          case 0x18: r = a & b; break;
          case 0x19: r = a | b; break;
          case 0x1a: r = a ^ b; break;
          case 0x1c: r = a << (b & 31); break;
          case 0x1d: r = a >> (b & 31); break;
          case 0x1e: r = (a >>> (b & 31)) | 0; break;
          case 0x20: r = a === b ? 1 : 0; break;
          case 0x21: r = a < b ? 1 : 0; break;
          default: r = (a >>> 0) < (b >>> 0) ? 1 : 0;
        }
        s[sp - 2] = r;
        sp--;
        break;
      }
      case 0x30: case 0x31: case 0x32: {
        // The offset is signed and counted from the instruction after the jump.
        const rel = ((p[next - 2] | (p[next - 1] << 8)) << 16) >> 16;
        let jump = true;
        if (op !== 0x30) {
          const v = s[--sp];
          jump = op === 0x31 ? v === 0 : v !== 0;
        }
        if (jump) pc = next + rel;
        break;
      }
      case 0x33:
        if (cp === 64) return stop(7);
        k[cp++] = next;
        pc = p[next - 2] | (p[next - 1] << 8);
        break;
      case 0x34:
        if (cp === 0) return stop(8);
        pc = k[--cp];
        break;
      case 0x40: {
        const addr = s[sp - 1];
        if ((addr >>> 0) >= 256) return stop(6);
        s[sp - 1] = m[addr];
        break;
      }
      case 0x41: {
        const addr = s[sp - 1], v = s[sp - 2];
        if ((addr >>> 0) >= 256) return stop(6);
        sp -= 2;
        m[addr] = v;
        break;
      }
      case 0x50: {
        const text = String(s[--sp]);
        if (ol + text.length > 65536) return stop(11);
        for (let i = 0; i < text.length; i++) o[ol++] = text.charCodeAt(i);
        break;
      }
      case 0x51:
        if (ol >= 65536) return stop(11);
        o[ol++] = s[--sp] & 255;
        break;
      case 0x52: s[sp++] = ip < inp.length ? inp[ip++] : -1; break;
      // 0x53 EXIT: only the low byte of the value is the exit code.
      default: return stop(s[--sp] & 255);
    }
  }
}

for (const entry of readdirSync(IN, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const src = `${IN}/${entry.name}`;
  const prog = readFileSync(`${src}/program.bin`);
  let input: Uint8Array;
  try {
    input = readFileSync(`${src}/input.bin`);
  } catch {
    input = new Uint8Array(0);
  }
  const [stdout, code] = vm(prog, input);
  const dst = `${OUT}/${entry.name}`;
  mkdirSync(dst, { recursive: true });
  writeFileSync(`${dst}/stdout.bin`, stdout);
  writeFileSync(`${dst}/exit.txt`, `${code}\n`);
}
