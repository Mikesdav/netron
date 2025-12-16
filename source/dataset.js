const dataset = {};

dataset.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (!stream) {
            return null;
        }
        const text = await context.read('text');
        const parsed = dataset.Reader.open(text);
        if (!parsed) {
            return null;
        }
        return context.set(parsed.format, parsed);
    }

    async open(context) {
        return new dataset.Model(context.value);
    }
};

dataset.Reader = class {

    static open(content) {
        if (!content || !content.length) {
            return null;
        }
        const data = dataset.Reader._parse(content);
        if (!data) {
            return null;
        }
        const names = dataset.Reader._names(data);
        const splits = dataset.Reader._splits(data);
        if (names.length === 0 && splits.length === 0) {
            return null;
        }
        const format = names.length > 0 || dataset.Reader._isYolo(data) ? 'YOLO Dataset' : 'Dataset';
        const metadata = dataset.Reader._metadata(data);
        return { format, data, names, splits, metadata };
    }

    static _parse(content) {
        const text = content.replace(/^\uFEFF/, '');
        const json = dataset.Reader._tryJSON(text);
        if (json) {
            return json;
        }
        return dataset.Reader._parseYaml(text);
    }

    static _tryJSON(text) {
        try {
            return JSON.parse(text);
        } catch {
            return null;
        }
    }

    static _parseYaml(text) {
        const lines = text.split(/\r?\n/);
        const root = {};
        let current = null;
        let currentIndent = 0;
        for (const line of lines) {
            const index = line.indexOf('#');
            const cleaned = (index >= 0 ? line.substring(0, index) : line).replace(/\s+$/, '');
            if (!cleaned.trim()) {
                continue;
            }
            if (!cleaned.startsWith(' ') && !cleaned.startsWith('\t')) {
                const match = cleaned.match(/^([^:]+):\s*(.*)$/);
                if (!match) {
                    continue;
                }
                const key = match[1].trim();
                const value = match[2];
                current = key;
                currentIndent = line.search(/\S/);
                if (value === '') {
                    root[key] = {};
                    if (key === 'names') {
                        root[key] = [];
                    }
                } else {
                    root[key] = dataset.Reader._value(value);
                }
                continue;
            }
            if (!current) {
                continue;
            }
            const indent = line.search(/\S/);
            if (indent <= currentIndent) {
                current = null;
                continue;
            }
            if (current === 'names') {
                const array = root.names;
                const mapItem = cleaned.match(/^\s*(\d+)\s*:\s*(.+)$/);
                if (mapItem) {
                    array[parseInt(mapItem[1], 10)] = dataset.Reader._value(mapItem[2]);
                    continue;
                }
                const listItem = cleaned.match(/^\s*-\s*(.+)$/);
                if (listItem) {
                    array.push(dataset.Reader._value(listItem[1]));
                }
                continue;
            }
            const target = root[current];
            if (target && typeof target === 'object' && !Array.isArray(target)) {
                const match = cleaned.match(/^\s*([^:]+):\s*(.*)$/);
                if (match) {
                    target[match[1].trim()] = dataset.Reader._value(match[2]);
                }
            }
        }
        return Object.keys(root).length > 0 ? root : null;
    }

    static _value(text) {
        const value = text.trim();
        if (value === '' || value === '~') {
            return null;
        }
        if (value === 'true') {
            return true;
        }
        if (value === 'false') {
            return false;
        }
        if (!Number.isNaN(Number(value))) {
            return Number(value);
        }
        if (value.startsWith('[') && value.endsWith(']')) {
            const items = value.substring(1, value.length - 1).split(',').map((item) => dataset.Reader._value(item));
            return items;
        }
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            return value.substring(1, value.length - 1);
        }
        return value;
    }

    static _names(data) {
        if (Array.isArray(data.names)) {
            return data.names.map((item, index) => item === undefined ? index.toString() : item).filter((item) => item !== null);
        }
        if (data.names && typeof data.names === 'object') {
            const entries = Object.entries(data.names).filter(([, value]) => value !== null && value !== undefined);
            entries.sort((a, b) => Number(a[0]) - Number(b[0]));
            return entries.map(([, value]) => value);
        }
        return [];
    }

    static _splits(data) {
        const keys = ['train', 'val', 'validation', 'test', 'predict'];
        const list = [];
        for (const key of keys) {
            if (data[key]) {
                list.push({ name: key, path: data[key] });
            }
        }
        return list;
    }

    static _metadata(data) {
        const list = [];
        const keys = ['path', 'download', 'roboflow', 'license', 'version', 'url', 'description', 'dataset'];
        for (const key of keys) {
            if (data[key] !== undefined && data[key] !== null) {
                list.push({ name: key, value: data[key] });
            }
        }
        const remaining = Object.entries(data).filter(([key]) => !keys.includes(key) && key !== 'names' && key !== 'nc' && !['train', 'val', 'validation', 'test', 'predict'].includes(key));
        for (const [name, value] of remaining) {
            list.push({ name, value });
        }
        return list;
    }

    static _isYolo(data) {
        return data.path !== undefined || data.nc !== undefined || data.roboflow !== undefined;
    }
};

dataset.Model = class {

    constructor(data) {
        this.format = data.format;
        this.metadata = dataset.Model._metadata(data);
        this.modules = [new dataset.Module(data)];
    }

    static _metadata(data) {
        const list = [];
        if (data.names.length > 0) {
            list.push({ name: 'classes', value: data.names.length });
        }
        const entry = data.metadata.find((item) => item.name === 'path');
        if (entry) {
            list.push({ name: 'path', value: entry.value });
        }
        const version = data.metadata.find((item) => item.name === 'version');
        if (version) {
            list.push({ name: 'version', value: version.value });
        }
        return list;
    }
};

dataset.Module = class {

    constructor(data) {
        this.name = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        if (data.names.length > 0) {
            this.nodes.push(new dataset.Node({ name: 'classes', type: 'Classes', attributes: [
                { name: 'count', value: data.names.length },
                { name: 'names', value: data.names }
            ] }));
        }
        for (const split of data.splits) {
            this.nodes.push(new dataset.Node({ name: split.name, type: 'Split', attributes: [
                { name: 'path', value: split.path }
            ] }));
        }
        if (data.metadata.length > 0) {
            this.nodes.push(new dataset.Node({ name: 'metadata', type: 'Metadata', attributes: data.metadata }));
        }
    }
};

dataset.Node = class {

    constructor(data) {
        this.name = data.name || '';
        this.type = { name: data.type || 'Object' };
        this.inputs = [];
        this.outputs = [];
        this.attributes = (data.attributes || []).map((attribute) => new dataset.Attribute(attribute));
    }
};

dataset.Attribute = class {

    constructor(attribute) {
        this.name = attribute.name || '';
        this.type = attribute.type || null;
        this.value = attribute.value;
    }
};

dataset.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Dataset Error';
    }
};

export { dataset as default };
export const ModelFactory = dataset.ModelFactory;
